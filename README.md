# wg-vanity

Generate WireGuard keypairs whose Base64 public key contains a chosen string
near the beginning.

[![CI](https://github.com/Harry-Chen/wg-vanity/actions/workflows/ci.yml/badge.svg)](https://github.com/Harry-Chen/wg-vanity/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/wg-vanity.svg)](https://crates.io/crates/wg-vanity)
[![Docs.rs](https://docs.rs/wg-vanity/badge.svg)](https://docs.rs/wg-vanity)
[![License](https://img.shields.io/crates/l/wg-vanity.svg)](LICENSE)

WireGuard public keys are difficult to distinguish at a glance. This tool
generates valid Curve25519 keypairs until the public key contains a memorable,
case-insensitive string within a configurable leading range.

This project is a modernized refresh of Brian Warner's original
[`wireguard-vanity-address`](https://github.com/warner/wireguard-vanity-address).
It preserves the original CPU key search while updating the Rust codebase and
adding bounded searches, CUDA and multi-GPU acceleration, and MPI scaling.

## CPU usage

Install the published CPU version:

```bash
cargo install wg-vanity
wg-vanity dave
```

The search uses all CPU cores. It continues until interrupted, printing every
matching keypair it finds. Searches can also be bounded:

```bash
# Search at most one million candidates.
wg-vanity dave --trials 1000000

# Search for at most 30 seconds.
wg-vanity dave --duration 30

# Match within the first 16 Base64 characters.
wg-vanity dave --in 16
```

The private value in a result is suitable for the `PrivateKey` field of a
WireGuard interface. Keep it secret.

## CUDA

The optional CUDA backend generates candidates, computes X25519, encodes the
public keys, and matches them on the GPU:

```bash
cargo build --release --features cuda --bin wg-vanity-cuda
./target/release/wg-vanity-cuda dave
```

The CUDA binary uses all visible GPUs by default. Use `--gpus N` to select a
smaller number, or restrict visibility with `CUDA_VISIBLE_DEVICES`. It reports
the estimated search space and, after measuring the hardware, the expected
time to find a match. `--trials`, `--duration`, `--batch`, and `--in` provide
the corresponding search controls.

A CUDA toolkit is required at build time. `CUDA_HOME` and `CUDA_ARCH` can be
used when the toolkit or target architecture cannot be detected from the
environment. See [docs/CUDA.md](docs/CUDA.md) for implementation and benchmark
details.

## MPI

MPI support is independent of CUDA. Build the CPU search with an installed MPI
implementation and launch it with the MPI launcher available on the system:

```bash
cargo build --release --features mpi --bin wg-vanity
mpiexec -n 8 ./target/release/wg-vanity dave
```

For distributed GPU search, enable both features:

```bash
cargo build --release --features cuda,mpi \
  --bin wg-vanity-cuda
mpiexec -n 2 ./target/release/wg-vanity-cuda dave
```

Each CPU rank uses its local CPU cores. Each CUDA rank uses all GPUs visible to
that rank unless `--gpus N` is specified. Rank 0 reports aggregate work and
throughput. GPU buffers are not passed through MPI, so CUDA-aware MPI is not
required.

## Search cost

For letters, case-insensitive Base64 matching adds roughly a factor of 32 for
every additional character. Allowing several possible starting offsets
improves the chance proportionally. Four or five characters are usually enough
to distinguish a small set of WireGuard peers; long prefixes become expensive
quickly.

Searches are memoryless and use fresh random private keys. Stopping and
restarting does not require a checkpoint, although work already performed is
not retained.

## Benchmarks

Run the Criterion CPU microbenchmarks with:

```bash
cargo bench
```

For bounded end-to-end throughput measurements, use `wg-vanity-benchmark` and
select the `cpu` or `cuda` backend.

## License

This project is distributed under the [MIT License](LICENSE).
