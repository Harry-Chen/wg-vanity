# [X86][SLP] Unprofitable AVX-512 vectorization on znver4

LLVM's SLP vectorizer makes a chained five-limb point-arithmetic benchmark
about 6% slower on Zen 4. The vectorized `point_add` uses ZMM registers,
masked broadcasts, and `vinserti32x4`/`vinserti64x4` repacking. Disabling SLP
or AVX-512 restores the scalar performance.

This was reduced from the serial fixed-base Curve25519 path in
`curve25519-dalek`. The reproducer has no external dependencies.

This reproducer deliberately isolates the point-addition SLP tree. The full
application has two LLVM SLP regressions and a separate rustc codegen-unit
partitioning effect. The latter is not an LLVM optimizer bug and is not
claimed to be reproduced by this file.

## Environment

- AMD EPYC 9654 (Zen 4)
- rustc 1.98.0, LLVM 22.1.8
- Linux x86-64

LLVM 23.1.0 from rustc nightly 1.100.0-nightly still reproduces the regression,
although the slowdown is smaller.

## Scope and issue ownership

The full application measurements decompose the regression as follows:

| Configuration | Throughput |
| --- | ---: |
| `native` (`znver4`) | 54.71 K keys/s |
| `native`, curve25519-dalek serial backend forced | 57.30 K keys/s |
| `native`, `-avx512f` | 64.30 K keys/s |

There are three relevant effects:

1. The point-addition SLP transformation reproduced here is an LLVM X86 cost
   model issue. It is legal code generation, but unprofitable on Zen 4.
2. A hotter constant-time fixed-base lookup has another unprofitable LLVM SLP
   transformation. The ZMM form uses `vpermq`, `vpermt2q`, and `vpternlogq`
   with stack spill/reload traffic between table entries. Compared with the
   AVX2 form, it retires effectively the same number of instructions but uses
   12.7% more L1D accesses and has 32.6% more L1D load misses.
3. Enabling curve25519-dalek's otherwise unused AVX-512 IFMA backend changes
   rustc's codegen-unit partitioning. `AffineNielsPoint::conditional_assign`
   and `LookupTable::select` then land in different LLVM modules, and the
   inline remark reports that the callee definition is unavailable. LLVM
   cannot inline a definition that is absent from its module without LTO.

The third effect is therefore best investigated as a rustc codegen-unit
partitioning and cross-CGU inlining issue, not as part of this LLVM report. It
is not yet clear whether rustc would consider that sensitivity a compiler bug
or an accepted multiple-CGU tradeoff; it needs a separate reduced reproducer.
Compiling the unused IFMA backend is the trigger, not evidence that IFMA code
is executed. An explicit inline attribute in curve25519-dalek or isolating the
optional backend may mitigate it, but neither is a substitute for fixing the
LLVM cost model.

Forcing the serial backend recovers about 4.7% (54.71 to 57.30 K keys/s).
Disabling AVX-512 while retaining SLP then recovers another 12.2% (57.30 to
64.30 K keys/s). These factors compose to the measured 17.5% end-to-end gain.
The 6.4% result below is smaller because this reproducer contains only the
point-addition tree; it does not contain the hotter lookup or the CGU effect.

The practical workaround addresses both compiler effects:

```bash
RUSTFLAGS="-C target-cpu=native -C target-feature=-avx512f" cargo build --release
```

It prevents the unused IFMA backend from changing the CGU layout and makes
LLVM choose the faster AVX2 SLP representation. Globally disabling SLP is a
worse workaround because it also removes profitable AVX2 vectorization.

## Reproduction

Use the attached `repro.rs`:

```bash
rustc --edition=2024 -O -C target-cpu=znver3 repro.rs -o repro-znver3
rustc --edition=2024 -O -C target-cpu=znver4 repro.rs -o repro-znver4
rustc --edition=2024 -O -C target-cpu=znver4 \
  -C target-feature=-avx512f repro.rs -o repro-no-avx512
rustc --edition=2024 -O -C target-cpu=znver4 \
  -C no-vectorize-slp repro.rs -o repro-no-slp

taskset -c 0 setarch x86_64 -R ./repro-znver3
taskset -c 0 setarch x86_64 -R ./repro-znver4
taskset -c 0 setarch x86_64 -R ./repro-no-avx512
taskset -c 0 setarch x86_64 -R ./repro-no-slp
```

Median of five runs on an otherwise idle node:

| Configuration | Throughput | Cycles | Instructions | IPC |
| --- | ---: | ---: | ---: | ---: |
| `znver3` | 8.83 M steps/s | 20.92 B | 72.75 B | 3.48 |
| `znver4` | 8.31 M steps/s | 22.26 B | 72.80 B | 3.27 |
| `znver4 -avx512f` | 8.83 M steps/s | 20.94 B | 72.75 B | 3.47 |
| `znver4`, no SLP | 8.82 M steps/s | 20.93 B | 72.75 B | 3.48 |

The result checksum is identical. Instruction counts are effectively equal,
but the SLP-vectorized version takes 6.4% more cycles. Disabling the loop
vectorizer does not change the result.

## Code generation

`point_add` is 0xc19 bytes with SLP and contains sequences such as:

```asm
vpbroadcastq    ..., %zmm...
vinserti64x4    ..., %zmm...
vinserti32x4    ..., %zmm...
vpaddq          ..., %zmm...
```

With `-C no-vectorize-slp` or `-C target-feature=-avx512f`, it is 0xbc5 bytes
and contains no ZMM operations. The latter is byte-for-byte the same size as
the `znver3` function.

Optimization remarks attribute the vectorization to the four five-limb stores
that construct `CompletedPoint`:

```text
Stores SLP vectorized with cost -3 and with tree size 22
Stores SLP vectorized with cost -3 and with tree size 28
Stores SLP vectorized with cost -7 and with tree size 12
```

The trees look locally profitable, but their packed results are immediately
consumed by the next point-arithmetic step. The extra packing and shuffle
latency is therefore on a loop-carried dependency chain.

## Expected behavior

The `znver4` cost model should reject this SLP transformation, or choose a
narrower representation that avoids the ZMM packing chain. Targeting Zen 4
should not be slower than `znver3` for this workload.

The likely LLVM fix belongs in X86 TTI/ScheduleModel or SLP profitability:
the cost needs to account for 512-bit permutation and repacking latency,
memory operands and spill pressure, and the dependency chain between adjacent
point operations. The fixed-base lookup should be reduced separately before
filing because this reproducer covers only the point-addition tree.

Possibly related, but with different transformations: #91370 and #87640.
