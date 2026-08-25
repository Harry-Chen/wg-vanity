//! CPU and optional CUDA primitives for searching WireGuard vanity keypairs.
//!
//! Most users should use the `wg-vanity` command-line programs. This library
//! exposes the single-candidate CPU operation and, with the `cuda` feature,
//! the reusable GPU batch searcher.

#![warn(missing_docs)]

use base64::{Engine as _, engine::general_purpose::STANDARD};
use regex::bytes::Regex;
use x25519_dalek::{PublicKey, StaticSecret};

/// Pattern syntax used by a key search.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PatternKind {
    /// Match a literal string.
    Literal,
    /// Match a glob with `*` (zero or more characters) and `?` (one character).
    Glob,
    /// Match a regular expression using the Rust `regex` engine.
    Regex,
}

/// A compiled search pattern shared by the CPU and CUDA frontends.
#[derive(Clone, Debug)]
pub struct SearchPattern {
    kind: PatternKind,
    bytes: Vec<u8>,
    regex: Option<Regex>,
    case_sensitive: bool,
}

/// The fixed-width pattern representation accepted by the CUDA matcher.
#[derive(Clone, Debug)]
pub struct GpuPattern {
    /// `0` for literal matching, `1` for glob matching.
    pub mode: u32,
    /// Pattern bytes, normalized when case-insensitive matching is enabled.
    pub bytes: Vec<u8>,
    /// Whether ASCII letter case must be preserved.
    pub case_sensitive: bool,
}

impl SearchPattern {
    /// Compiles `text` with the requested syntax and case behavior.
    pub fn new(text: &str, kind: PatternKind, case_sensitive: bool) -> Result<Self, String> {
        let bytes = if case_sensitive {
            text.as_bytes().to_vec()
        } else {
            text.bytes().map(|byte| byte.to_ascii_lowercase()).collect()
        };
        let regex = if kind == PatternKind::Regex {
            Some(
                regex::bytes::RegexBuilder::new(text)
                    .case_insensitive(!case_sensitive)
                    .build()
                    .map_err(|error| format!("invalid regular expression: {error}"))?,
            )
        } else {
            None
        };
        Ok(Self {
            kind,
            bytes,
            regex,
            case_sensitive,
        })
    }

    /// Returns the selected pattern syntax.
    pub fn kind(&self) -> PatternKind {
        self.kind
    }

    /// Returns the original matching text length in bytes.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Returns whether the pattern text is empty.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Returns whether this pattern can be evaluated by the CUDA matcher.
    pub fn gpu_pattern(&self) -> Option<GpuPattern> {
        match self.kind {
            PatternKind::Literal => Some(GpuPattern {
                mode: 0,
                bytes: self.bytes.clone(),
                case_sensitive: self.case_sensitive,
            }),
            PatternKind::Glob => Some(GpuPattern {
                mode: 1,
                bytes: self.bytes.clone(),
                case_sensitive: self.case_sensitive,
            }),
            PatternKind::Regex => None,
        }
    }

    /// Tests the pattern against any substring within `start..end`.
    pub fn is_match(&self, encoded: &[u8], start: usize, end: usize) -> bool {
        if start > end || end > encoded.len() {
            return false;
        }
        let haystack = &encoded[start..end];
        match self.kind {
            PatternKind::Literal => {
                let normalized = if self.case_sensitive {
                    haystack.to_vec()
                } else {
                    haystack
                        .iter()
                        .map(|byte| byte.to_ascii_lowercase())
                        .collect()
                };
                normalized
                    .windows(self.bytes.len())
                    .any(|window| window == self.bytes.as_slice())
            }
            PatternKind::Glob => {
                let normalized = if self.case_sensitive {
                    haystack.to_vec()
                } else {
                    haystack
                        .iter()
                        .map(|byte| byte.to_ascii_lowercase())
                        .collect()
                };
                glob_matches(&normalized, &self.bytes)
            }
            PatternKind::Regex => self
                .regex
                .as_ref()
                .is_some_and(|regex| regex.is_match(haystack)),
        }
    }
}

fn glob_matches(haystack: &[u8], pattern: &[u8]) -> bool {
    (0..=haystack.len()).any(|start| glob_matches_at(&haystack[start..], pattern))
}

fn glob_matches_at(text: &[u8], pattern: &[u8]) -> bool {
    let mut text_index = 0;
    let mut pattern_index = 0;
    let mut star_index = None;
    let mut star_text_index = 0;
    while text_index < text.len() {
        if pattern_index == pattern.len() {
            return true;
        }
        if pattern_index < pattern.len()
            && pattern[pattern_index] != b'*'
            && (pattern[pattern_index] == b'?' || pattern[pattern_index] == text[text_index])
        {
            pattern_index += 1;
            text_index += 1;
        } else if pattern_index < pattern.len() && pattern[pattern_index] == b'*' {
            star_index = Some(pattern_index);
            pattern_index += 1;
            star_text_index = text_index;
        } else if let Some(star) = star_index {
            pattern_index = star + 1;
            star_text_index += 1;
            text_index = star_text_index;
        } else {
            return false;
        }
    }
    while pattern_index < pattern.len() && pattern[pattern_index] == b'*' {
        pattern_index += 1;
    }
    pattern_index == pattern.len()
}

#[cfg(feature = "cuda")]
/// CUDA batch search support.
pub mod cuda;

/// Generates one keypair and returns it when its public key matches `prefix`.
///
/// Matching is performed against `public_key[start..end]` after that range and
/// prefix are converted to ASCII lowercase. A match returns
/// `(private_key, public_key)`, both Base64 encoded.
///
/// # Panics
///
/// Panics when `start..end` is not a valid range within the 44-character
/// Base64 public key.
pub fn trial(prefix: &str, start: usize, end: usize) -> Option<(String, String)> {
    assert!(start <= end && end <= 44, "invalid public-key range");
    let pattern = SearchPattern::new(prefix, PatternKind::Literal, false)
        .expect("literal patterns are always valid");
    trial_pattern(&pattern, start, end)
}

/// Generates one keypair and returns it when its public key matches `pattern`.
pub fn trial_pattern(
    pattern: &SearchPattern,
    start: usize,
    end: usize,
) -> Option<(String, String)> {
    let private = StaticSecret::random();
    let public = PublicKey::from(&private);
    let public_b64 = STANDARD.encode(public.as_bytes());
    if pattern.is_match(public_b64.as_bytes(), start, end) {
        let private_b64 = STANDARD.encode(private.to_bytes());
        Some((private_b64, public_b64))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_case_behavior() {
        let insensitive = SearchPattern::new("Ab", PatternKind::Literal, false).unwrap();
        let sensitive = SearchPattern::new("Ab", PatternKind::Literal, true).unwrap();
        assert!(insensitive.is_match(b"xxaBzz", 0, 6));
        assert!(!sensitive.is_match(b"xxaBzz", 0, 6));
        assert!(sensitive.is_match(b"xxAbzz", 0, 6));
    }

    #[test]
    fn glob_supports_wildcards_and_range() {
        let pattern = SearchPattern::new("a*?z", PatternKind::Glob, false).unwrap();
        assert!(pattern.is_match(b"xxAbcZyy", 0, 8));
        assert!(!pattern.is_match(b"xxAbcZyy", 0, 5));
    }

    #[test]
    fn regex_uses_rust_regex_engine() {
        let pattern = SearchPattern::new("a[0-9]+z", PatternKind::Regex, true).unwrap();
        assert!(pattern.is_match(b"xxa123zyy", 0, 9));
        assert!(pattern.gpu_pattern().is_none());
    }
}
