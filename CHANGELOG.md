# Changelog

All notable changes to `wg-vanity` are documented here.

## Unreleased

- Add literal, glob (`*` and `?`), regular-expression, and optional
  case-sensitive matching to the CPU CLI.
- Add CUDA literal/glob matching and report regex mode as CPU-only.
- Stop CPU searches after the first match, including coordinated MPI ranks.

## 0.5.1 - 2026-08-25

- Detect the lowest compute capability reported by visible GPUs when building
  the CUDA PTX, with `CUDA_ARCH` as an explicit override.
- Fall back to `compute_80` when no GPU is available during the build.
- Add a GitHub Actions release workflow for crates.io publishing on version tags.

## 0.5.0 - 2026-08-24

- Refresh the original `wireguard-vanity-address` project as `wg-vanity`, with
  updated Rust dependencies and public API documentation.
- Add bounded CPU and CUDA searches with candidate, duration, and batch limits,
  plus search-space and time estimates.
- Add an optimized CUDA backend with multi-GPU selection and reusable buffers.
- Add MPI support independently of CUDA for CPU searches and for distributed
  GPU searches; CUDA buffers are never passed through MPI.
- Remove the per-candidate atomic counter from the CPU search path.
- Add Criterion benchmarks and GitHub Actions coverage for CPU, MPI, CUDA, and
  the minimum supported Rust version.
