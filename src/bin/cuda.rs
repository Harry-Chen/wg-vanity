use clap::Parser;
#[cfg(feature = "mpi")]
use std::env;
use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier, OnceLock, mpsc};
use std::thread;
use std::time::{Duration, Instant};
use wg_vanity::cuda::GpuSearcher;

#[cfg(feature = "mpi")]
use mpi::traits::*;

fn expected_candidates(prefix: &str, end: usize) -> f64 {
    let len = prefix.len();
    let offsets = end.saturating_sub(len).saturating_add(1);
    if len == 0 || offsets == 0 {
        return f64::INFINITY;
    }
    let casefolded_letters = prefix.bytes().filter(|c| c.is_ascii_alphabetic()).count();
    let alphabet = 64_f64.powi(len as i32);
    alphabet / (offsets as f64 * 2_f64.powi(casefolded_letters as i32))
}

fn format_count(count: f64) -> String {
    if count.is_finite() && count < 1_000_000.0 {
        format!("{count:.0}")
    } else if count.is_finite() {
        format!("{count:.3e}")
    } else {
        "unbounded".to_string()
    }
}

fn format_duration(seconds: f64) -> String {
    if !seconds.is_finite() {
        return "unbounded".to_string();
    }
    if seconds >= 3600.0 {
        format!("{:.2} h", seconds / 3600.0)
    } else if seconds >= 60.0 {
        format!("{:.1} min", seconds / 60.0)
    } else if seconds >= 1.0 {
        format!("{seconds:.1} s")
    } else {
        format!("{:.1} ms", seconds * 1e3)
    }
}

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

    /// Number of visible GPUs to use (defaults to all visible GPUs).
    #[arg(long, value_name = "COUNT")]
    gpus: Option<usize>,

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

enum WorkerMessage {
    Batch {
        attempts: u64,
        candidate: Option<(String, String)>,
    },
    Ready,
    Done,
    Error(String),
}

#[cfg(feature = "mpi")]
const MPI_FOUND_TAG: i32 = 701;
#[cfg(feature = "mpi")]
const MPI_STOP_TAG: i32 = 702;
#[cfg(feature = "mpi")]
const MPI_SUMMARY_TAG: i32 = 703;
#[cfg(feature = "mpi")]
const MPI_KEY_TAG: i32 = 704;

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "mpi")]
    let universe = mpi::initialize().ok_or("MPI was already initialized")?;
    #[cfg(feature = "mpi")]
    let world = universe.world();
    #[cfg(feature = "mpi")]
    let rank = world.rank();
    #[cfg(feature = "mpi")]
    let local_world = world.split_shared(0);
    #[cfg(feature = "mpi")]
    let local_rank = local_world.rank() as usize;
    #[cfg(feature = "mpi")]
    let local_size = local_world.size() as usize;
    #[cfg(not(feature = "mpi"))]
    let rank = 0;
    let root = rank == 0;

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

    let expected = expected_candidates(&prefix, end);
    if root {
        println!(
            "expected work: about {} candidates (case-insensitive estimate) to find one match",
            format_count(expected)
        );
    }

    let visible_gpus = GpuSearcher::device_count()
        .map_err(|e| format!("CUDA device enumeration failed: {e:?}"))?;
    if visible_gpus == 0 {
        return Err("CUDA reported zero visible devices".into());
    }
    #[cfg(feature = "mpi")]
    let partition_local_gpus = local_size > 1
        && env::var_os("SLURM_GPUS_PER_TASK").is_none()
        && visible_gpus.is_multiple_of(local_size);
    #[cfg(feature = "mpi")]
    let available_gpus = if partition_local_gpus {
        visible_gpus / local_size
    } else {
        visible_gpus
    };
    #[cfg(not(feature = "mpi"))]
    let available_gpus = visible_gpus;

    if args.gpus == Some(0) {
        return Err("--gpus must be greater than zero".into());
    }
    if args.gpus.is_some_and(|count| count > available_gpus) {
        return Err(format!("--gpus exceeds available CUDA devices ({available_gpus})").into());
    }
    let workers = args.gpus.unwrap_or(available_gpus);
    #[cfg(feature = "mpi")]
    let gpu_offset = if partition_local_gpus {
        local_rank * available_gpus
    } else {
        0
    };
    if root {
        #[cfg(feature = "mpi")]
        println!(
            "using {workers} CUDA device(s) per MPI rank ({})",
            world.size()
        );
        #[cfg(not(feature = "mpi"))]
        println!("using {workers} of {visible_gpus} visible CUDA device(s)");
    }

    let target_batches = args.batches.unwrap_or(u64::MAX);
    let estimate_batches = args
        .trials
        .map(|limit| limit.div_ceil(args.batch))
        .unwrap_or(workers as u64)
        .min(target_batches)
        .max(1);
    let stop = Arc::new(AtomicBool::new(false));
    let start_gate = Arc::new(Barrier::new(workers + 1));
    let start_time = Arc::new(OnceLock::new());
    let (tx, rx) = mpsc::channel();
    let mut handles = Vec::with_capacity(workers);
    for device in 0..workers {
        #[cfg(feature = "mpi")]
        let device = gpu_offset + device;
        let tx = tx.clone();
        let stop = Arc::clone(&stop);
        let start_gate = Arc::clone(&start_gate);
        let start_time = Arc::clone(&start_time);
        let prefix = prefix.clone();
        let duration = args.duration;
        let trials = args.trials;
        let batch = args.batch;
        handles.push(thread::spawn(move || {
            let gpu = match GpuSearcher::new_on_device(device) {
                Ok(gpu) => Some(gpu),
                Err(error) => {
                    stop.store(true, Ordering::Relaxed);
                    let _ = tx.send(WorkerMessage::Error(format!(
                        "CUDA device {device} initialization failed: {error:?}"
                    )));
                    None
                }
            };
            let _ = tx.send(WorkerMessage::Ready);
            start_gate.wait();
            let started = *start_time.get().expect("main thread sets start time");
            let deadline =
                duration.map(|seconds| started + std::time::Duration::from_secs_f64(seconds));
            let Some(mut gpu) = gpu else {
                let _ = tx.send(WorkerMessage::Done);
                return;
            };
            let mut batch_index = device as u64;
            while batch_index < target_batches
                && !stop.load(Ordering::Relaxed)
                && deadline.is_none_or(|limit| Instant::now() < limit)
            {
                let base_counter = batch_index.saturating_mul(batch);
                let count = trials
                    .map(|limit| batch.min(limit.saturating_sub(base_counter)))
                    .unwrap_or(batch);
                if count == 0 {
                    break;
                }
                match gpu.search_batch(&prefix, 0, end, count, base_counter) {
                    Ok(result) => {
                        if result.candidate.is_some() {
                            stop.store(true, Ordering::Relaxed);
                        }
                        if tx
                            .send(WorkerMessage::Batch {
                                attempts: result.attempts,
                                candidate: result.candidate,
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(error) => {
                        stop.store(true, Ordering::Relaxed);
                        let _ = tx.send(WorkerMessage::Error(format!(
                            "CUDA device {device} launch failed: {error:?}"
                        )));
                        break;
                    }
                }
                batch_index = batch_index.saturating_add(workers as u64);
            }
            let _ = tx.send(WorkerMessage::Done);
        }));
    }
    drop(tx);

    let mut ready_workers = 0usize;
    let mut startup_error = None;
    while ready_workers < workers {
        match rx.recv() {
            Ok(WorkerMessage::Ready) => ready_workers += 1,
            Ok(WorkerMessage::Error(message)) => {
                startup_error.get_or_insert(message);
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }
    let started = Instant::now();
    let _ = start_time.set(started);
    start_gate.wait();

    let mut total = 0u64;
    let mut completed_batches = 0u64;
    let mut done_workers = 0usize;
    let mut candidate = None;
    let mut error = startup_error;
    let mut estimate_printed = false;
    let mut last_progress = started;
    #[cfg(feature = "mpi")]
    let mut mpi_stop_sent = false;
    while done_workers < workers {
        #[cfg(feature = "mpi")]
        {
            if root && !mpi_stop_sent {
                if let Some(status) = world.any_process().immediate_probe_with_tag(MPI_FOUND_TAG) {
                    let _ = world
                        .process_at_rank(status.source_rank())
                        .receive_with_tag::<i32>(MPI_FOUND_TAG);
                    stop.store(true, Ordering::Relaxed);
                    for destination in 1..world.size() {
                        world
                            .process_at_rank(destination)
                            .send_with_tag(&1i32, MPI_STOP_TAG);
                    }
                    mpi_stop_sent = true;
                }
            } else if !root
                && !stop.load(Ordering::Relaxed)
                && let Some(status) = world.any_process().immediate_probe_with_tag(MPI_STOP_TAG)
            {
                let _ = world
                    .process_at_rank(status.source_rank())
                    .receive_with_tag::<i32>(MPI_STOP_TAG);
                stop.store(true, Ordering::Relaxed);
            }
        }

        match rx.recv_timeout(Duration::from_millis(10)) {
            Ok(WorkerMessage::Batch {
                attempts,
                candidate: found,
            }) => {
                total = total.saturating_add(attempts);
                completed_batches += 1;
                if candidate.is_none() {
                    candidate = found;
                    if candidate.is_some() {
                        #[cfg(feature = "mpi")]
                        if !root {
                            world.process_at_rank(0).send_with_tag(&1i32, MPI_FOUND_TAG);
                        } else if !mpi_stop_sent {
                            stop.store(true, Ordering::Relaxed);
                            for destination in 1..world.size() {
                                world
                                    .process_at_rank(destination)
                                    .send_with_tag(&1i32, MPI_STOP_TAG);
                            }
                            mpi_stop_sent = true;
                        }
                    }
                }
                if !estimate_printed && completed_batches >= estimate_batches {
                    let local_rate =
                        total as f64 / started.elapsed().as_secs_f64().max(f64::MIN_POSITIVE);
                    if root {
                        #[cfg(feature = "mpi")]
                        let rate = local_rate * world.size() as f64;
                        #[cfg(not(feature = "mpi"))]
                        let rate = local_rate;
                        println!(
                            "measured {:.3e} keys/s; estimated time to a match: {}",
                            rate,
                            format_duration(expected / rate)
                        );
                    }
                    estimate_printed = true;
                }
                if args.batches.is_none() && last_progress.elapsed() >= Duration::from_secs(5) {
                    if root {
                        #[cfg(feature = "mpi")]
                        let displayed_total = total.saturating_mul(world.size() as u64);
                        #[cfg(not(feature = "mpi"))]
                        let displayed_total = total;
                        let rate = displayed_total as f64 / started.elapsed().as_secs_f64();
                        println!("searched about {displayed_total} candidates ({rate:.3e} keys/s)");
                    }
                    last_progress = Instant::now();
                }
            }
            Ok(WorkerMessage::Done) => done_workers += 1,
            Ok(WorkerMessage::Ready) => {}
            Ok(WorkerMessage::Error(message)) => {
                error.get_or_insert(message);
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
        }
    }
    let finished = started.elapsed().as_secs_f64();
    for handle in handles {
        let _ = handle.join();
    }
    if let Some(error) = error {
        return Err(error.into());
    }
    #[cfg(feature = "mpi")]
    {
        let mut local_key = [0u8; 89];
        if let Some((private, public)) = &candidate {
            local_key[0] = 1;
            local_key[1..45].copy_from_slice(private.as_bytes());
            local_key[45..89].copy_from_slice(public.as_bytes());
        }
        if root {
            let mut global_total = total;
            let mut global_elapsed = finished;
            let mut found = candidate;
            for source in 1..world.size() {
                let (summary, _) = world
                    .process_at_rank(source)
                    .receive_vec_with_tag::<u64>(MPI_SUMMARY_TAG);
                if summary.len() != 2 {
                    return Err(format!("MPI rank {source} sent an invalid summary").into());
                }
                global_total = global_total.saturating_add(summary[0]);
                global_elapsed = global_elapsed.max(f64::from_bits(summary[1]));

                let (key, _) = world
                    .process_at_rank(source)
                    .receive_vec_with_tag::<u8>(MPI_KEY_TAG);
                if key.len() != local_key.len() {
                    return Err(format!("MPI rank {source} sent an invalid key buffer").into());
                }
                if found.is_none() && key[0] != 0 {
                    found = Some((
                        String::from_utf8_lossy(&key[1..45]).into_owned(),
                        String::from_utf8_lossy(&key[45..89]).into_owned(),
                    ));
                }
            }
            if let Some((private, public)) = found {
                println!("private {private}  public {public}");
            }
            println!(
                "MPI: {} ranks, {global_total} candidates in {global_elapsed:.3}s ({:.3e} keys/s)",
                world.size(),
                global_total as f64 / global_elapsed
            );
        } else {
            world
                .process_at_rank(0)
                .send_with_tag(&[total, finished.to_bits()], MPI_SUMMARY_TAG);
            world
                .process_at_rank(0)
                .send_with_tag(&local_key, MPI_KEY_TAG);
        }
        Ok(())
    }

    #[cfg(not(feature = "mpi"))]
    {
        if let Some((private, public)) = candidate {
            println!("private {private}  public {public}");
            println!("searched {total} candidates in {finished:.3}s");
            return Ok(());
        }
        println!(
            "CUDA benchmark: {total} candidates in {finished:.3}s ({:.3e} keys/s)",
            total as f64 / finished
        );
    }
    #[cfg(not(feature = "mpi"))]
    Ok(())
}
