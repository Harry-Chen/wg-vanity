# Performance Notes

## End-to-end throughput

Environment: one RTX 5090 (SM 12.0, 32 GiB), two AMD EPYC 9654 CPUs,
NVIDIA driver 610.57.04, CUDA Toolkit 13.3, and rustc 1.98.0.

Median throughput from three runs on 2026-08-25:

| Backend | Candidates | Time | Throughput |
| --- | ---: | ---: | ---: |
| CPU + Rayon (384 threads) | 32,000,000 | 2.959 s | 10.81 M keys/s |
| CUDA + Rust host (1 GPU) | 64,000,000 | 1.321 s | 48.45 M keys/s |

The CUDA measurement includes per-batch copies, kernel launch, and
synchronization, but excludes initial context creation. One GPU is about 4.48x
faster than the full CPU node for this workload.

## CUDA kernel

The initial 16-limb kernel used 255 registers per thread and generated 48B of
spill stores and loads. The current 5x51-bit implementation uses 128 registers
per thread with no spills. Nsight Compute reported about 80.6% compute
throughput and 0.3% memory throughput. Dedicated squaring improved throughput
by roughly 15% over generic multiply-based squaring.

## CPU target tuning

`-C target-cpu=native` is not automatically faster for this workload. A
single-thread, CPU-pinned comparison on the EPYC 9654 host produced:

| Rust target | Throughput | Cycles/candidate | IPC |
| --- | ---: | ---: | ---: |
| default `x86-64` | 62.54 K keys/s | 58,877 | 3.51 |
| `x86-64-v3` | 62.83 K keys/s | 58,596 | 2.72 |
| `znver3` | 65.05 K keys/s | 56,889 | 2.81 |
| `x86-64-v4` | 61.79 K keys/s | 59,886 | 2.77 |
| `native` (`znver4`) | 54.96 K keys/s | 66,981 | 2.41 |

The clock remained near 3.7 GHz in every run. `perf` recorded about 48 billion
retired instructions for both tuned targets, while the AMD backend-stall event
rose from 36.0 billion with `znver3` to 53.2 billion with `znver4`. The
regression is therefore not AVX-512 downclocking or simply more instructions.

Public-key generation uses x25519-dalek's precomputed fixed-base path, which
is implemented with the serial 5x51-bit field backend. The AVX-512 IFMA backend
is compiled by a native build but is not selected for this path. Disabling the
precomputed table does exercise IFMA, but the resulting variable-base path
measured only 55.89 K keys/s because it loses the faster fixed-base algorithm.

The `znver4` build also makes different inlining and code-layout decisions for
the constant-time lookup path. Raising LLVM's global inline threshold recovers
most, but not all, of the regression; forcing individual helpers to inline
causes excessive code growth and is not a suitable fix. For current releases,
the portable default remains recommended. On this CPU, `znver3` is the fastest
tested target. A proper compiler fix needs a reduced `znver3` versus `znver4`
reproducer for LLVM's cost model; useful AVX-512 acceleration requires a
fixed-base or multi-candidate batch implementation rather than switching to
the existing variable-base backend.
