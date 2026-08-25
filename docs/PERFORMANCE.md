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
| `native`, `-avx512f` | 64.30 K keys/s |

Each row is the median of five 300,000-candidate runs with ASLR disabled. The
last configuration retains Zen 4, BMI2, and AVX2 tuning while disabling
AVX-512 code generation. It is 17.5% faster than the unmodified native build
and slightly faster than `znver3`.

Public-key generation uses x25519-dalek's precomputed fixed-base path, which
is implemented with the serial 5x51-bit field backend. The AVX-512 IFMA backend
is compiled by a native build but is not selected for this path. Disabling the
precomputed table does exercise IFMA, but the resulting variable-base path
measured only 55.89 K keys/s because it loses the faster fixed-base algorithm.

The regression comes from LLVM's SLP vectorizer, not from the IFMA backend.
For `znver4`, LLVM combines the five-limb additions and subtractions at the end
of a point addition into ZMM operations. The resulting code contains masked
broadcasts and `vinserti32x4`/`vinserti64x4` repacking on the dependency path.
Disabling loop vectorization has no effect; either `-C no-vectorize-slp` or
`-C target-feature=-avx512f` removes this code and restores throughput.

The standalone reproducer is in
[`contrib/llvm-znver4-slp-repro/repro.rs`](../contrib/llvm-znver4-slp-repro/repro.rs).
On the same node with rustc 1.98.0 and LLVM 22.1.8:

| Reproducer target | Throughput | Cycles | Instructions | IPC |
| --- | ---: | ---: | ---: | ---: |
| `znver3` | 8.83 M steps/s | 20.92 B | 72.75 B | 3.48 |
| `znver4` | 8.31 M steps/s | 22.26 B | 72.80 B | 3.27 |
| `znver4`, `-avx512f` | 8.83 M steps/s | 20.94 B | 72.75 B | 3.47 |
| `znver4`, no SLP | 8.82 M steps/s | 20.93 B | 72.75 B | 3.48 |

The reduced benchmark has effectively the same retired instruction count, but
the vectorized build needs 6.4% more cycles. LLVM's optimization remarks score
the relevant SLP store trees as locally profitable, but do not capture their
cost on the loop-carried point dependency. LLVM 23.1.0 still reproduces the
issue, although with a smaller slowdown.

Compile and run the reduced case with ASLR disabled for stable layout:

```bash
rustc --edition=2024 -O -C target-cpu=znver3 repro.rs -o repro-znver3
rustc --edition=2024 -O -C target-cpu=znver4 repro.rs -o repro-znver4
rustc --edition=2024 -O -C target-cpu=znver4 \
  -C no-vectorize-slp repro.rs -o repro-no-slp
setarch x86_64 -R ./repro-znver3
setarch x86_64 -R ./repro-znver4
setarch x86_64 -R ./repro-no-slp
```

Until LLVM's cost model is corrected, a native Zen 4 build can use:

```bash
RUSTFLAGS="-C target-cpu=native -C target-feature=-avx512f" cargo build --release
```

The portable build remains the default because Cargo packages cannot assume
the build host and runtime host are the same machine.
