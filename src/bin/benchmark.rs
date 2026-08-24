use clap::Parser;
use std::time::Instant;
use wireguard_vanity_lib::trial;

#[derive(Debug, Parser)]
#[command(author, version, about = "Compare CPU and CUDA vanity-key throughput")]
struct Args {
    /// Backend to benchmark: cpu, or cuda when built with --features cuda.
    #[arg(long, default_value = "cpu")]
    backend: String,

    /// Prefix used only for the match predicate; a long prefix avoids result allocation.
    #[arg(long, default_value = "zzzzzzzzzz")]
    prefix: String,

    /// Number of candidates for CPU, or total candidates for CUDA.
    #[arg(long, default_value_t = 1_000_000)]
    trials: u64,

    /// Number of candidates per CUDA kernel launch.
    #[arg(long, default_value_t = 1 << 20)]
    batch: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    match args.backend.as_str() {
        "cpu" => benchmark_cpu(&args),
        "cuda" => benchmark_cuda(&args),
        other => Err(format!("unknown backend {other:?}; use cpu or cuda").into()),
    }
}

fn benchmark_cpu(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    use rayon::prelude::*;
    let end = 44.min(args.prefix.len().max(10));
    let start = Instant::now();
    let matches = (0..args.trials)
        .into_par_iter()
        .filter_map(|_| trial(&args.prefix, 0, end))
        .count();
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "CPU: {} candidates in {elapsed:.3}s ({:.3e} keys/s, {matches} matches)",
        args.trials,
        args.trials as f64 / elapsed
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn benchmark_cuda(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    use wireguard_vanity_lib::cuda::GpuSearcher;
    let end = 44.min(args.prefix.len().max(10));
    let mut gpu = GpuSearcher::new().map_err(|e| format!("CUDA initialization failed: {e:?}"))?;
    let start = Instant::now();
    let mut done = 0u64;
    let mut counter = 0u64;
    while done < args.trials {
        let count = args.batch.min(args.trials - done);
        let result = gpu
            .search_batch(&args.prefix, 0, end, count, counter)
            .map_err(|e| format!("CUDA launch failed: {e:?}"))?;
        done += result.attempts;
        counter = counter.saturating_add(result.attempts);
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "CUDA: {done} candidates in {elapsed:.3}s ({:.3e} keys/s)",
        done as f64 / elapsed
    );
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn benchmark_cuda(_args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    Err("CUDA backend is not compiled; rebuild with --features cuda".into())
}
