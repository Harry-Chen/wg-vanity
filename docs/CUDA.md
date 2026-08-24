# CUDA Backend and Rust CUDA Evaluation

## Implementation

The optional `cuda` feature adds a CUDA backend. Rust owns the CUDA context,
memory management, kernel launch, result encoding, and command-line interface.
The device kernel is in `cuda/vanity_x25519.cu`; `build.rs` compiles it to PTX
with `nvcc`, and `cudarc` loads the PTX at runtime.

Each device thread processes one candidate:

1. The host generates a 256-bit CSPRNG seed.
2. The device expands `seed + counter` with ChaCha20 into a private key.
3. The device computes X25519 with a Montgomery ladder.
4. The device encodes the public key as Base64 and checks the requested range.
5. `atomicCAS` returns the first matching private/public key pair.

The device does not run `OsRng` or `x25519-dalek`. Each batch gets a fresh host
CSPRNG seed; ChaCha20 provides a strong device-side stream without transferring
one random key per candidate.

## Build and run

CPU:

```bash
cargo test
cargo run --release --bin wireguard-vanity-address -- dave
```

CUDA on RTX 5090/Blackwell:

```bash
CUDA_HOME=/usr/local/cuda-13.3 \
  cargo build --release --features cuda --bin wireguard-vanity-address-cuda

CUDA_HOME=/usr/local/cuda-13.3 \
  ./target/release/wireguard-vanity-address-cuda dave --batch 1048576
```

CPU and CUDA searches run until a match is found or interrupted with `Ctrl-C`.
Use limits for bounded runs:

```bash
# CPU: at most 10M candidates.
cargo run --release --bin wireguard-vanity-address -- dave --trials 10000000

# CUDA: at most 60 seconds. Duration is checked between batches.
CUDA_HOME=/usr/local/cuda-13.3 \
  ./target/release/wireguard-vanity-address-cuda dave \
  --duration 60 --batch 1048576

# The first reached limit wins.
CUDA_HOME=/usr/local/cuda-13.3 \
  ./target/release/wireguard-vanity-address-cuda dave \
  --trials 10000000 --duration 60 --batches 100
```

`cuda` is a Cargo feature, not a runtime switch. Without `--features cuda`,
`src/cuda.rs` and the CUDA binary are not compiled. With the feature enabled,
`build.rs` invokes `nvcc` and writes `OUT_DIR/vanity_x25519.ptx`; the Rust host
loads it with `cudarc`. The default target is `compute_120` and can be changed:

```bash
CUDA_HOME=/usr/local/cuda-13.3 CUDA_ARCH=compute_120 \
  cargo build --release --features cuda --bin wireguard-vanity-address-cuda
```

Throughput-only benchmark:

```bash
CUDA_HOME=/usr/local/cuda-13.3 \
  ./target/release/wireguard-vanity-benchmark --backend cpu --trials 8000000

CUDA_HOME=/usr/local/cuda-13.3 \
  ./target/release/wireguard-vanity-benchmark --backend cuda \
  --trials 8000000 --batch 8000000
```

## RTX 5090 results

Environment: RTX 5090 (SM 12.0, 32 GiB), NVIDIA driver 610.57.04, CUDA
Toolkit 13.3.

One long-batch run on 2026-08-24:

| Backend | Candidates | Time | Throughput |
| --- | ---: | ---: | ---: |
| CPU + Rayon | 16,000,000 | 3.517 s | 4.55 M keys/s |
| CUDA + Rust host | 16,000,000 | 0.337 s | 47.54 M keys/s |

The CUDA number includes per-batch copies, kernel launch, synchronization, and
counter readback, but excludes initial CUDA context creation. This is about
10.45x faster than the CPU path.

The initial 16-limb kernel used 255 registers per thread and generated 48B of
spill stores and loads. The current version uses 5x51-bit field limbs,
dedicated squaring, and addition-chain inversion: 128 registers per thread and
zero spills. Nsight Compute reported about 80.6% compute throughput and 0.3%
memory throughput. Dedicated squaring added roughly 15% throughput over the
generic multiply-based square.

Long-running searches should use `--duration`, `--trials`, or `--batches`.
The `--batch` value controls both throughput and the granularity at which a
duration limit is observed. Reusing device buffers avoids repeated allocation
and free operations between batches.

## Rust-native CUDA evaluation

NVIDIA's `cuda-oxide` was tested in a temporary checkout with current nightly,
Clang 19, and LLVM tools. `cargo-oxide doctor` passed for CUDA 13.3, the RTX
5090, libNVVM, nvJitLink, LLVM, and Clang.

The repository's pinned `nightly-2026-04-03` is not available from the current
rustup mirror. With nightly 1.100.0, the `rustc-codegen-cuda` backend failed on
26 `rustc_public` API compatibility errors, including
`CrateDefType::ty_with_args` and newer MIR `Rvalue::Use` fields. The production
kernel was not migrated to an unbuildable oxide backend.

The production boundary is Rust host code plus an auditable CUDA kernel. A
future oxide port should first implement an independent X25519 correctness
kernel and cross-check it against RFC 7748 and `x25519-dalek`.

## Key security notes

The GPU search temporarily stores private keys in device global memory and
returns the first match. Do not run it on an untrusted or multi-tenant GPU.
After a match, write the key into the WireGuard configuration and terminate the
process. The CUDA path generates fresh keys and never accepts an existing
private key as a command-line argument.
