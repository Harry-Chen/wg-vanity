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
single-thread, CPU-pinned comparison on an otherwise idle EPYC 9654 node
produced:

| Rust target | Throughput |
| --- | ---: |
| default `x86-64` | 62.17 K keys/s |
| `znver3` | 64.17 K keys/s |
| `native` (`znver4`) | 54.71 K keys/s |
| `native`, forced dalek serial backend | 57.30 K keys/s |
| `native`, `-avx512f` | 64.30 K keys/s |

Each row is the median of five 300,000-candidate runs with ASLR disabled. The
last configuration retains Zen 4, BMI2, and AVX2 tuning while disabling
AVX-512 code generation. It is 17.5% faster than the unmodified native build
and slightly faster than `znver3`.

The regression combines unprofitable LLVM AVX-512 SLP transformations with a
separate rustc codegen-unit partitioning effect triggered by an unused IFMA
backend. The profiler evidence, issue ownership, compiler remarks, and reduced
benchmark are documented in
[`contrib/llvm-znver4-slp-repro/LLVM-ISSUE.md`](../contrib/llvm-znver4-slp-repro/LLVM-ISSUE.md).

Until LLVM's cost model is corrected, a native Zen 4 build can use:

```bash
RUSTFLAGS="-C target-cpu=native -C target-feature=-avx512f" cargo build --release
```

The portable build remains the default because Cargo packages cannot assume
the build host and runtime host are the same machine.
