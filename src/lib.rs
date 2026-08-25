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
    #[cfg(feature = "cuda")]
    source: String,
    bytes: Vec<u8>,
    regex: Option<Regex>,
    case_sensitive: bool,
}

#[cfg(feature = "cuda")]
/// A pattern prepared for the CUDA matcher.
#[derive(Clone, Debug)]
pub enum GpuPattern {
    /// Literal matching data.
    Literal {
        /// Pattern bytes, normalized when case-insensitive matching is enabled.
        bytes: Vec<u8>,
        /// Whether ASCII letter case must be preserved.
        case_sensitive: bool,
    },
    /// Basic glob matching data.
    Glob {
        /// Pattern bytes, normalized when case-insensitive matching is enabled.
        bytes: Vec<u8>,
        /// Whether ASCII letter case must be preserved.
        case_sensitive: bool,
    },
    /// A compact regex DFA shared by GPU workers.
    Regex(std::sync::Arc<crate::regex_dfa::GpuRegexDfa>),
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
            #[cfg(feature = "cuda")]
            source: text.to_string(),
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

    /// Compiles this pattern into the representation used by the CUDA matcher.
    #[cfg(feature = "cuda")]
    pub fn gpu_pattern(&self) -> Result<GpuPattern, crate::regex_dfa::GpuRegexError> {
        match self.kind {
            PatternKind::Literal => Ok(GpuPattern::Literal {
                bytes: self.bytes.clone(),
                case_sensitive: self.case_sensitive,
            }),
            PatternKind::Glob => Ok(GpuPattern::Glob {
                bytes: self.bytes.clone(),
                case_sensitive: self.case_sensitive,
            }),
            PatternKind::Regex => Ok(GpuPattern::Regex(std::sync::Arc::new(
                crate::regex_dfa::GpuRegexDfa::compile(&self.source, self.case_sensitive)?,
            ))),
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
                if self.bytes.is_empty() {
                    true
                } else if self.case_sensitive {
                    haystack
                        .windows(self.bytes.len())
                        .any(|window| window == self.bytes.as_slice())
                } else {
                    haystack.windows(self.bytes.len()).any(|window| {
                        window
                            .iter()
                            .zip(&self.bytes)
                            .all(|(input, pattern)| input.to_ascii_lowercase() == *pattern)
                    })
                }
            }
            PatternKind::Glob => glob_matches(haystack, &self.bytes, self.case_sensitive),
            PatternKind::Regex => self
                .regex
                .as_ref()
                .is_some_and(|regex| regex.is_match(haystack)),
        }
    }
}

fn glob_matches(haystack: &[u8], pattern: &[u8], case_sensitive: bool) -> bool {
    (0..=haystack.len()).any(|start| glob_matches_at(&haystack[start..], pattern, case_sensitive))
}

fn glob_matches_at(text: &[u8], pattern: &[u8], case_sensitive: bool) -> bool {
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
            && (pattern[pattern_index] == b'?' || {
                let input = text[text_index];
                let expected = pattern[pattern_index];
                if case_sensitive {
                    input == expected
                } else {
                    input.to_ascii_lowercase() == expected
                }
            })
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
/// Host-side CUDA regex compilation and compact DFA simulation.
pub mod regex_dfa;
#[cfg(feature = "cuda")]
pub use regex_dfa::{GpuRegexDfa, GpuRegexError};

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
    let mut public_b64 = [0u8; 44];
    let public_len = STANDARD
        .encode_slice(public.as_bytes(), &mut public_b64)
        .expect("32-byte public keys always fit in a 44-byte Base64 buffer");
    if pattern.is_match(&public_b64[..public_len], start, end) {
        let private_b64 = STANDARD.encode(private.to_bytes());
        Some((
            private_b64,
            String::from_utf8(public_b64[..public_len].to_vec()).unwrap(),
        ))
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

        let sensitive = SearchPattern::new("A*Z", PatternKind::Glob, true).unwrap();
        assert!(sensitive.is_match(b"xxAbcZyy", 0, 8));
        assert!(!sensitive.is_match(b"xxabczyy", 0, 8));
    }

    #[test]
    fn regex_uses_rust_regex_engine() {
        let pattern = SearchPattern::new("a[0-9]+z", PatternKind::Regex, true).unwrap();
        assert!(pattern.is_match(b"xxa123zyy", 0, 9));
        #[cfg(feature = "cuda")]
        assert!(pattern.gpu_pattern().is_ok());
    }
}
