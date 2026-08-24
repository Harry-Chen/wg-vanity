use clap::Parser;
use std::error::Error;
use std::time::Instant;
use wireguard_vanity_lib::cuda::GpuSearcher;

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "CUDA-accelerated WireGuard vanity key search"
)]
struct Args {
    /// String to find near the start of the public key.
    name: String,

    /// Search range in Base64 characters (defaults to 10, or len+10 for long names).
    #[arg(long = "in")]
    range: Option<usize>,

    /// Number of candidates per kernel launch.
    #[arg(long, default_value_t = 1 << 20)]
    batch: u64,

    /// Stop after this many candidates. The first limit reached wins.
    #[arg(long, value_name = "COUNT")]
    trials: Option<u64>,

    /// Stop after this many seconds. The first limit reached wins.
    #[arg(long, value_name = "SECONDS")]
    duration: Option<f64>,

    /// Stop after this many kernel launches. The first limit reached wins.
    #[arg(long)]
    batches: Option<u64>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let prefix = args.name.to_ascii_lowercase();
    let end = 44.min(args.range.unwrap_or_else(|| {
        if prefix.len() <= 10 {
            10
        } else {
            prefix.len() + 10
        }
    }));
    if prefix.is_empty() || prefix.len() > end {
        return Err(format!("prefix must fit in the selected range (0..{end})").into());
    }
    if args
        .duration
        .is_some_and(|seconds| !seconds.is_finite() || seconds <= 0.0)
    {
        return Err("--duration must be a finite positive number".into());
    }
    if args.batch == 0 {
        return Err("--batch must be greater than zero".into());
    }

    let mut gpu = GpuSearcher::new().map_err(|e| format!("CUDA initialization failed: {e:?}"))?;
    let mut counter = 0u64;
    let mut total = 0u64;
    let started = Instant::now();
    let deadline = args
        .duration
        .map(|seconds| started + std::time::Duration::from_secs_f64(seconds));
    let target_batches = args.batches.unwrap_or(u64::MAX);
    let mut batch_index = 0u64;
    loop {
        if batch_index >= target_batches
            || args.trials.is_some_and(|limit| total >= limit)
            || deadline.is_some_and(|limit| Instant::now() >= limit)
        {
            break;
        }
        let count = args
            .trials
            .map(|limit| args.batch.min(limit.saturating_sub(total)))
            .unwrap_or(args.batch);
        if count == 0 {
            break;
        }
        let result = gpu
            .search_batch(&prefix, 0, end, count, counter)
            .map_err(|e| format!("CUDA launch failed: {e:?}"))?;
        counter = counter.saturating_add(result.attempts);
        total = total.saturating_add(result.attempts);
        if let Some((private, public)) = result.candidate {
            println!("private {private}  public {public}");
            println!(
                "searched {total} candidates in {:.3}s",
                started.elapsed().as_secs_f64()
            );
            return Ok(());
        }
        if args.batches.is_none() && (batch_index + 1) % 10 == 0 {
            let rate = total as f64 / started.elapsed().as_secs_f64();
            println!("searched {total} candidates ({rate:.3e} keys/s)");
        }
        batch_index += 1;
    }

    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "CUDA benchmark: {total} candidates in {elapsed:.3}s ({:.3e} keys/s)",
        total as f64 / elapsed
    );
    Ok(())
}
