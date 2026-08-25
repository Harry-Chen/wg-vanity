use std::error::Error;
use std::fmt;
use std::io::{self, Write};
use std::time::{Duration, Instant, SystemTime};

use clap::Parser;
use rayon::prelude::*;
use wg_vanity::{PatternKind, SearchPattern, trial_pattern};

#[cfg(feature = "mpi")]
use mpi::collective::SystemOperation;
#[cfg(feature = "mpi")]
use mpi::traits::*;

#[cfg(feature = "mpi")]
const MPI_SUMMARY_TAG: i32 = 801;

fn estimate_one_trial(pattern: &SearchPattern) -> Duration {
    let start = SystemTime::now();
    const COUNT: u32 = 100;
    (0..COUNT).for_each(|_| {
        trial_pattern(pattern, 0, 10);
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
        private_b64,
        public_b64
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
    /// NAME must match within the first RANGE chars of the public key.
    #[arg(long = "in")]
    range: Option<usize>,

    /// Interpret NAME as a glob (`*` and `?`) instead of a literal.
    #[arg(long, conflicts_with = "regex")]
    glob: bool,

    /// Interpret NAME as a regular expression (CPU search only).
    #[arg(long, conflicts_with = "glob")]
    regex: bool,

    /// Preserve ASCII letter case while matching.
    #[arg(long)]
    case_sensitive: bool,

    /// Stop after this many candidate keys. Combined with --duration, the first limit wins.
    #[arg(long, value_name = "COUNT")]
    trials: Option<u64>,

    /// Stop after this many seconds. Combined with --trials, the first limit wins.
    #[arg(long, value_name = "SECONDS")]
    duration: Option<f64>,

    /// Literal, glob, or regular expression to find near the start of the public key.
    name: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "mpi")]
    let universe = mpi::initialize().ok_or("MPI was already initialized")?;
    #[cfg(feature = "mpi")]
    let world = universe.world();
    #[cfg(feature = "mpi")]
    let rank = world.rank();
    #[cfg(feature = "mpi")]
    let root = rank == 0;
    #[cfg(not(feature = "mpi"))]
    let root = true;

    let args = Args::parse();
    let kind = if args.glob {
        PatternKind::Glob
    } else if args.regex {
        PatternKind::Regex
    } else {
        PatternKind::Literal
    };
    let pattern = SearchPattern::new(&args.name, kind, args.case_sensitive)?;
    let len = pattern.len();
    let end: usize = 44.min(args.range.unwrap_or_else(|| {
        if kind == PatternKind::Literal && len > 10 {
            len + 10
        } else {
            10
        }
    }));
    if end == 0 || (kind == PatternKind::Literal && end < len) {
        return Err(ParseError(format!("search range {} is invalid for this pattern", end)).into());
    }
    if args
        .duration
        .is_some_and(|seconds| !seconds.is_finite() || seconds <= 0.0)
    {
        return Err(ParseError("--duration must be a finite positive number".into()).into());
    }

    if root {
        println!("searching for '{}' in pubkey[0..{}]", args.name, end);
    }

    if kind == PatternKind::Literal {
        let offsets = 1 + end - len;
        let casefolded_letters = if args.case_sensitive {
            0
        } else {
            args.name
                .bytes()
                .filter(|c| c.is_ascii_alphabetic())
                .count()
        };
        let trials_per_key =
            64_f64.powi(len as i32) / (offsets as f64 * 2_f64.powi(casefolded_letters as i32));
        let trials_description = if trials_per_key < 1_000_000.0 {
            format!("{trials_per_key:.0}")
        } else {
            format!("{trials_per_key:.3e}")
        };
        if root {
            println!(
                "one of every {} keys should match{}",
                trials_description,
                if args.case_sensitive {
                    " (case-sensitive)"
                } else {
                    " (case-insensitive)"
                }
            );
        }
        if trials_per_key < 2_f64.powi(32) {
            let est = estimate_one_trial(&pattern);
            #[cfg(feature = "mpi")]
            let parallelism = num_cpus::get().saturating_mul(world.size() as usize);
            #[cfg(not(feature = "mpi"))]
            let parallelism = num_cpus::get();
            let spk = duration_to_f64(est) * trials_per_key / parallelism as f64;
            let kps = 1.0 / spk;
            if root {
                #[cfg(feature = "mpi")]
                println!(
                    "one trial takes {}, {} CPU cores/rank, {} MPI ranks",
                    format_time(duration_to_f64(est)),
                    num_cpus::get(),
                    world.size()
                );
                #[cfg(not(feature = "mpi"))]
                println!(
                    "one trial takes {}, CPU cores available: {}",
                    format_time(duration_to_f64(est)),
                    num_cpus::get()
                );
                println!(
                    "est yield: {} per key, {}",
                    format_time(spk),
                    format_rate(kps)
                );
            }
        }
    } else if root {
        println!("search-space estimate unavailable for glob/regex patterns");
    }

    let started = Instant::now();
    let deadline = args
        .duration
        .map(|seconds| started + Duration::from_secs_f64(seconds));
    let global_max_trials = args.trials.unwrap_or(u64::MAX);
    #[cfg(feature = "mpi")]
    let max_trials = if global_max_trials == u64::MAX {
        u64::MAX
    } else {
        let ranks = world.size() as u64;
        global_max_trials / ranks + u64::from((rank as u64) < global_max_trials % ranks)
    };
    #[cfg(not(feature = "mpi"))]
    let max_trials = global_max_trials;
    if root {
        println!("searching until a match, a limit, or Ctrl-C");
    }

    const CPU_BATCH: u64 = 100_000;
    let mut attempted = 0u64;
    #[cfg(not(feature = "mpi"))]
    while attempted < max_trials && deadline.is_none_or(|limit| Instant::now() < limit) {
        let count = CPU_BATCH.min(max_trials - attempted);
        let found = (0..count)
            .into_par_iter()
            .find_map_any(|_| trial_pattern(&pattern, 0, end));
        attempted += count;
        if let Some(result) = found {
            print(result)?;
            break;
        }
    }
    #[cfg(feature = "mpi")]
    loop {
        let active = attempted < max_trials && deadline.is_none_or(|limit| Instant::now() < limit);
        let found = if active {
            let count = CPU_BATCH.min(max_trials - attempted);
            let found = (0..count)
                .into_par_iter()
                .find_map_any(|_| trial_pattern(&pattern, 0, end));
            attempted += count;
            found
        } else {
            None
        };

        let local_found = i32::from(found.is_some());
        let mut global_found = 0;
        world.all_reduce_into(&local_found, &mut global_found, SystemOperation::max());
        if global_found != 0 {
            let mut payload = [0u8; 89];
            if let Some((private, public)) = found {
                payload[0] = 1;
                payload[1..45].copy_from_slice(private.as_bytes());
                payload[45..89].copy_from_slice(public.as_bytes());
            }
            if root {
                let mut selected = if payload[0] == 1 {
                    Some((
                        String::from_utf8(payload[1..45].to_vec())?,
                        String::from_utf8(payload[45..89].to_vec())?,
                    ))
                } else {
                    None
                };
                for source in 1..world.size() {
                    let (received, _) = world.process_at_rank(source).receive_vec::<u8>();
                    if received.len() != 89 {
                        return Err(
                            format!("MPI rank {source} sent an invalid match payload").into()
                        );
                    }
                    if selected.is_none() && received[0] == 1 {
                        selected = Some((
                            String::from_utf8(received[1..45].to_vec())?,
                            String::from_utf8(received[45..89].to_vec())?,
                        ));
                    }
                }
                if let Some(result) = selected {
                    print(result)?;
                }
            } else {
                world.process_at_rank(0).send(&payload[..]);
            }
            break;
        }

        let local_done = i32::from(!active);
        let mut global_done = 0;
        world.all_reduce_into(&local_done, &mut global_done, SystemOperation::min());
        if global_done != 0 {
            break;
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    #[cfg(feature = "mpi")]
    if root {
        let mut global_attempted = attempted;
        let mut global_elapsed = elapsed;
        for source in 1..world.size() {
            let (summary, _) = world
                .process_at_rank(source)
                .receive_vec_with_tag::<u64>(MPI_SUMMARY_TAG);
            if summary.len() != 2 {
                return Err(format!("MPI rank {source} sent an invalid summary").into());
            }
            global_attempted = global_attempted.saturating_add(summary[0]);
            global_elapsed = global_elapsed.max(f64::from_bits(summary[1]));
        }
        println!(
            "MPI: {} ranks stopped after {} candidates in {:.3}s",
            world.size(),
            global_attempted,
            global_elapsed
        );
    } else {
        world
            .process_at_rank(0)
            .send_with_tag(&[attempted, elapsed.to_bits()], MPI_SUMMARY_TAG);
    }
    #[cfg(not(feature = "mpi"))]
    println!("stopped after {attempted} candidates in {elapsed:.3}s");
    Ok(())
}
