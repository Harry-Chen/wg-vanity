use base64::{Engine as _, engine::general_purpose::STANDARD};
use x25519_dalek::{PublicKey, StaticSecret};

#[cfg(feature = "cuda")]
pub mod cuda;

pub fn trial(prefix: &str, start: usize, end: usize) -> Option<(String, String)> {
    let private = StaticSecret::random();
    let public = PublicKey::from(&private);
    let public_b64 = STANDARD.encode(public.as_bytes());
    if public_b64[start..end]
        .to_ascii_lowercase()
        .contains(&prefix)
    {
        let private_b64 = STANDARD.encode(private.to_bytes());
        Some((private_b64, public_b64))
    } else {
        None
    }
}
