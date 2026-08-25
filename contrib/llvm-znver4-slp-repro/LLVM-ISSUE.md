# [X86][SLP] Unprofitable AVX-512 vectorization on znver4

LLVM's SLP vectorizer makes a chained five-limb point-arithmetic benchmark
about 6% slower on Zen 4. The vectorized `point_add` uses ZMM registers,
masked broadcasts, and `vinserti32x4`/`vinserti64x4` repacking. Disabling SLP
or AVX-512 restores the scalar performance.

This was reduced from the serial fixed-base Curve25519 path in
`curve25519-dalek`. The reproducer has no external dependencies.

This reproducer deliberately isolates the point-addition SLP tree. The larger
application-level regression also contains a separate SLP transformation in
the fixed-base lookup and a rustc codegen-unit partitioning effect; those are
not claimed to be reproduced by this file.

## Environment

- AMD EPYC 9654 (Zen 4)
- rustc 1.98.0, LLVM 22.1.8
- Linux x86-64

LLVM 23.1.0 from rustc nightly 1.100.0-nightly still reproduces the regression,
although the slowdown is smaller.

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

Possibly related, but with different transformations: #91370 and #87640.
