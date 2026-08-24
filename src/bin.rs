use std::error::Error;
use std::fmt;
use std::io::{self, Write};
use std::time::{Duration, Instant, SystemTime};

use clap::Parser;
use num_cpus;
use rayon::prelude::*;
use wireguard_vanity_lib::trial;

fn estimate_one_trial() -> Duration {
    let prefix = "prefix";
    let start = SystemTime::now();
    const COUNT: u32 = 100;
    (0..COUNT).for_each(|_| {
        trial(&prefix, 0, 10);
    });
    let elapsed = start.elapsed().unwrap();
    elapsed.checked_div(COUNT).unwrap()
}

fn duration_to_f64(d: Duration) -> f64 {
    (d.as_secs() as f64) + (f64::from(d.subsec_nanos()) * 1e-9)
}

fn format_time(t: f64) -> String {
    if t > 3600.0 {
        format!("{:.2} hours", t / 3600.0)
    } else if t > 60.0 {
        format!("{:.1} minutes", t / 60.0)
    } else if t > 1.0 {
        format!("{:.1} seconds", t)
    } else if t > 1e-3 {
        format!("{:.1} ms", t * 1e3)
    } else if t > 1e-6 {
        format!("{:.1} us", t * 1e6)
    } else if t > 1e-9 {
        format!("{:.1} ns", t * 1e9)
    } else {
        format!("{:.3} ps", t * 1e12)
    }
}

fn format_rate(rate: f64) -> String {
    if rate > 1e9 {
        format!("{:.2}e9 keys/s", rate / 1e9)
    } else if rate > 1e6 {
        format!("{:.2}e6 keys/s", rate / 1e6)
    } else if rate > 1e3 {
        format!("{:.2}e3 keys/s", rate / 1e3)
    } else if rate > 1e0 {
        format!("{:.2} keys/s", rate)
    } else if rate > 1e-3 {
        format!("{:.2}e-3 keys/s", rate * 1e3)
    } else if rate > 1e-6 {
        format!("{:.2}e-6 keys/s", rate * 1e6)
    } else if rate > 1e-9 {
        format!("{:.2}e-9 keys/s", rate * 1e9)
    } else {
        format!("{:.3}e-12 keys/s", rate * 1e12)
    }
}

fn print(res: (String, String)) -> Result<(), io::Error> {
    let (private_b64, public_b64) = res;
    writeln!(
        io::stdout(),
        "private {}  public {}",
        &private_b64,
        &public_b64
    )
}

#[derive(Debug)]
struct ParseError(String);
impl Error for ParseError {}
impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Finds WireGuard keypairs with a given string prefix"
)]
struct Args {
    /// NAME must be found within the first RANGE chars of the public key.
    #[arg(long = "in")]
    range: Option<usize>,

    /// Stop after this many candidate keys. Combined with --duration, the first limit wins.
    #[arg(long, value_name = "COUNT")]
    trials: Option<u64>,

    /// Stop after this many seconds. Combined with --trials, the first limit wins.
    #[arg(long, value_name = "SECONDS")]
    duration: Option<f64>,

    /// String to find near the start of the public key.
    name: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let prefix = args.name.to_ascii_lowercase();
    let len = prefix.len();
    let end: usize = 44.min(
        args.range
            .unwrap_or_else(|| if len <= 10 { 10 } else { len + 10 }),
    );
    if end < len {
        return Err(ParseError(format!("range {} is too short for len={}", end, len)).into());
    }
    if args
        .duration
        .is_some_and(|seconds| !seconds.is_finite() || seconds <= 0.0)
    {
        return Err(ParseError("--duration must be a finite positive number".into()).into());
    }

    let offsets: u64 = 44.min((1 + end - len) as u64);
    // todo: this is an approximation, offsets=2 != double the chances
    let mut num = offsets;
    let mut denom = 1u64;
    prefix.chars().for_each(|c| {
        if c.is_ascii_alphabetic() {
            num *= 2; // letters can match both uppercase and lowercase
        }
        denom *= 64; // base64
    });
    let trials_per_key = denom / num;

    println!(
        "searching for '{}' in pubkey[0..{}], one of every {} keys should match",
        &prefix, end, trials_per_key
    );

    // todo: dividing by num_cpus will overestimate performance when the
    // cores aren't actually distinct (hyperthreading?). My Core-i7 seems to
    // run at half the speed that this predicts.

    if trials_per_key < 2u64.pow(32) {
        let est = estimate_one_trial();
        println!(
            "one trial takes {}, CPU cores available: {}",
            format_time(duration_to_f64(est)),
            num_cpus::get()
        );
        let spk = duration_to_f64(
            est // sec/trial on one core
                .checked_div(num_cpus::get() as u32) // sec/trial with all cores
                .unwrap()
                .checked_mul(trials_per_key as u32) // sec/key (Duration)
                .unwrap(),
        );
        let kps = 1.0 / spk;
        println!(
            "est yield: {} per key, {}",
            format_time(spk),
            format_rate(kps)
        );
    }

    let started = Instant::now();
    let deadline = args
        .duration
        .map(|seconds| started + Duration::from_secs_f64(seconds));
    let max_trials = args.trials.unwrap_or(u64::MAX);
    println!("searching until a match, a limit, or Ctrl-C");

    const CPU_BATCH: u64 = 100_000;
    let mut attempted = 0u64;
    while attempted < max_trials && deadline.is_none_or(|limit| Instant::now() < limit) {
        let count = CPU_BATCH.min(max_trials - attempted);
        let matches: Vec<_> = (0..count)
            .into_par_iter()
            .map(|_| trial(&prefix, 0, end))
            .filter_map(|result| result)
            .collect();
        for result in matches {
            print(result)?;
        }
        attempted += count;
    }
    println!(
        "stopped after {} candidates in {:.3}s",
        attempted,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
