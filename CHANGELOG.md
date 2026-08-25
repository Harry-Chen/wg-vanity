# Changelog

All notable changes to `wg-vanity` are documented here.

## Unreleased

- Document and reduce an LLVM `znver4` SLP-vectorization regression in the
  serial X25519 fixed-base path, including a targeted AVX-512 workaround.

## 0.6.0 - 2026-08-25

- Add literal, glob (`*` and `?`), and regular-expression matching with
  optional case-sensitive behavior to the CPU and CUDA CLIs.
- Compile CUDA regexes into a bounded Base64 DFA, with EOI handling, compact
  transitions, explicit size limits, and one table upload per GPU.
- Add prepared GPU patterns, a streaming Base64 regex kernel, and CUDA matcher
  differential tests without transferring candidate strings to the host.
- Stop CPU searches after the first match, including coordinated MPI ranks.
- Add Criterion comparisons for literal, glob, and precompiled-regex matching.
- Avoid per-candidate CPU Base64 and case-folding allocations, and stop Rayon
  batch work after the first match is found.
- Optimize CUDA X25519 field arithmetic and streaming DFA matching while
  keeping the literal and glob kernels on their existing fast path.
- Document CUDA regex semantics, resource limits, and the GPU data path.

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
