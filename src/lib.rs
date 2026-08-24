//! CPU and optional CUDA primitives for searching WireGuard vanity keypairs.
//!
//! Most users should use the `wg-vanity` command-line programs. This library
//! exposes the single-candidate CPU operation and, with the `cuda` feature,
//! the reusable GPU batch searcher.

#![warn(missing_docs)]

use base64::{Engine as _, engine::general_purpose::STANDARD};
use x25519_dalek::{PublicKey, StaticSecret};

#[cfg(feature = "cuda")]
/// CUDA batch search support.
pub mod cuda;

/// Generates one keypair and returns it when its public key matches `prefix`.
///
/// Matching is performed against `public_key[start..end]` after that range is
/// converted to ASCII lowercase. Callers should therefore pass a lowercase
/// prefix. A match returns `(private_key, public_key)`, both Base64 encoded.
///
/// # Panics
///
/// Panics when `start..end` is not a valid range within the 44-character
/// Base64 public key.
pub fn trial(prefix: &str, start: usize, end: usize) -> Option<(String, String)> {
    let private = StaticSecret::random();
    let public = PublicKey::from(&private);
    let public_b64 = STANDARD.encode(public.as_bytes());
    if public_b64[start..end].to_ascii_lowercase().contains(prefix) {
        let private_b64 = STANDARD.encode(private.to_bytes());
        Some((private_b64, public_b64))
    } else {
        None
    }
}
