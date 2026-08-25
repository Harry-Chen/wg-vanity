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

Public-key generation uses x25519-dalek's precomputed fixed-base path, which
is implemented with the serial 5x51-bit field backend. The AVX-512 IFMA backend
is compiled by a native build but is not selected for this path. Disabling the
precomputed table does exercise IFMA, but the resulting variable-base path
measured only 55.89 K keys/s because it loses the faster fixed-base algorithm.

Two effects make up the regression. First, enabling the otherwise unused IFMA
backend changes rustc's codegen-unit partitioning. In the native build,
`AffineNielsPoint::conditional_assign` and `LookupTable::select` land in
different units; LLVM's inline remark says the callee definition is
unavailable. Forcing the dalek serial backend makes the helper available and
raises throughput from 54.71 to 57.30 K keys/s. This is a compilation-layout
effect, not runtime IFMA execution.

The larger effect is LLVM's SLP code for the constant-time fixed-base lookup.
The lookup carries 15 field limbs through eight table entries. The `znver4`
version uses `vpermq`, `vpermt2q`, and `vpternlogq` on ZMM registers and spills
the intermediate state to the stack between entries. The `-avx512f` build
keeps SLP enabled but uses a shorter-latency XMM/YMM representation. On Zen 4,
the two versions retire almost exactly 160 billion instructions for one
million candidates, but their hardware counters differ:

| Fixed-base lookup build | Cycles | IPC | L1D loads | L1D load misses |
| --- | ---: | ---: | ---: | ---: |
| `native`, forced serial | 64.44 B | 2.48 | 49.29 B | 130.0 M |
| `native`, `-avx512f` | 57.46 B | 2.79 | 43.74 B | 98.1 M |

Thus the ZMM build performs 12.7% more L1D accesses and has 32.6% more L1D
misses without reducing the retired instruction count. Its lookup alone takes
about 11.85 billion sampled cycles versus 3.97 billion for the AVX2 version.

There is also a smaller SLP issue in point addition. LLVM combines five-limb
additions and subtractions into masked broadcasts and
`vinserti32x4`/`vinserti64x4` repacking on a dependency path. Disabling loop
vectorization has no effect. Disabling SLP globally only raises full-program
throughput to 55.57 K keys/s because it removes both the harmful AVX-512 trees
and useful AVX2 SLP. Disabling AVX-512 is the better workaround because it
retains the useful vectorization.

The standalone reproducer is in
[`contrib/llvm-znver4-slp-repro/repro.rs`](../contrib/llvm-znver4-slp-repro/repro.rs).
On the same node with rustc 1.98.0 and LLVM 22.1.8:

| Reproducer target | Throughput | Cycles | Instructions | IPC |
| --- | ---: | ---: | ---: | ---: |
| `znver3` | 8.83 M steps/s | 20.92 B | 72.75 B | 3.48 |
| `znver4` | 8.31 M steps/s | 22.26 B | 72.80 B | 3.27 |
| `znver4`, `-avx512f` | 8.83 M steps/s | 20.94 B | 72.75 B | 3.47 |
| `znver4`, no SLP | 8.82 M steps/s | 20.93 B | 72.75 B | 3.48 |

The reduced benchmark isolates only the point-addition part of the regression.
It has effectively the same retired instruction count, but the vectorized
build needs 6.4% more cycles. LLVM's optimization remarks score the relevant
SLP store trees as locally profitable, but do not capture their cost on the
loop-carried point dependency. The full-program slowdown is larger because it
also includes the hotter lookup transformation and the codegen-unit effect
described above. LLVM 23.1.0 still reproduces the reduced issue, although with
a smaller slowdown.

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
