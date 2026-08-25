//! A bounded, Base64-specialized DFA for CUDA regex matching.

use std::collections::{HashMap, VecDeque};
use std::fmt;

use regex_automata::dfa::{Automaton, StartKind, dense};
use regex_automata::{Input, MatchKind};

/// A transition that accepts immediately.
pub const DFA_MATCH: u32 = 1 << 31;
/// A transition that cannot match.
pub const DFA_DEAD: u32 = 1 << 30;
/// Mask for a compact DFA state ID.
pub const DFA_STATE_MASK: u32 = (1 << 30) - 1;

const BASE64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const PATTERN_MAX_BYTES: usize = 4096;
const DETERMINIZE_SIZE_LIMIT: usize = 32 * 1024 * 1024;
const DFA_SIZE_LIMIT: usize = 32 * 1024 * 1024;
const GPU_STATE_LIMIT: usize = 4096;
const GPU_TABLE_LIMIT: usize = 1024 * 1024;

/// Errors raised while compiling a regex for the CUDA DFA backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuRegexError {
    /// The regex is empty; CUDA matching requires at least one input byte.
    EmptyPattern,
    /// The pattern exceeds the host-side input limit.
    PatternTooLong,
    /// The regex parser or DFA builder rejected the pattern.
    Compile(String),
    /// The compact DFA exceeds the state limit.
    TooManyStates {
        /// Number of compact states requested.
        states: usize,
        /// Maximum number of compact states.
        limit: usize,
    },
    /// The compact transition table exceeds the upload limit.
    TableTooLarge {
        /// Number of bytes requested.
        bytes: usize,
        /// Maximum table size.
        limit: usize,
    },
    /// A quit state was reachable while compacting the DFA.
    QuitStateReachable,
    /// The DFA contained an unsupported transition state.
    InvalidState,
    /// An input byte is not part of the Base64 alphabet.
    InvalidInput(u8),
}

impl fmt::Display for GpuRegexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPattern => write!(f, "empty regexes are not supported by CUDA"),
            Self::PatternTooLong => write!(f, "regex exceeds the {PATTERN_MAX_BYTES}-byte limit"),
            Self::Compile(error) => write!(f, "regex DFA compilation failed: {error}"),
            Self::TooManyStates { states, limit } => {
                write!(f, "compact DFA has {states} states; limit is {limit}")
            }
            Self::TableTooLarge { bytes, limit } => {
                write!(f, "compact DFA requires {bytes} bytes; limit is {limit}")
            }
            Self::QuitStateReachable => write!(f, "DFA quit state is reachable"),
            Self::InvalidState => write!(f, "DFA contains an invalid state"),
            Self::InvalidInput(byte) => write!(f, "input byte 0x{byte:02x} is not Base64"),
        }
    }
}

impl std::error::Error for GpuRegexError {}

/// A compact forward DFA operating on the 64-character Base64 alphabet.
#[derive(Clone, Debug)]
pub struct GpuRegexDfa {
    /// State-major transitions indexed as `state * 64 + sextet`.
    pub transitions: Vec<u32>,
    /// Transition used for the final Base64 padding byte `=`.
    pub equals: Vec<u32>,
    /// Whether the EOI transition from each state accepts.
    pub eoi_match: Vec<u8>,
    /// Number of compact states. State zero is always the start state.
    pub state_count: u32,
}

impl GpuRegexDfa {
    /// Compiles `pattern` with the requested ASCII case behavior.
    pub fn compile(pattern: &str, case_sensitive: bool) -> Result<Self, GpuRegexError> {
        if pattern.is_empty() {
            return Err(GpuRegexError::EmptyPattern);
        }
        if pattern.len() > PATTERN_MAX_BYTES {
            return Err(GpuRegexError::PatternTooLong);
        }

        let syntax = regex_automata::util::syntax::Config::new()
            .case_insensitive(!case_sensitive)
            .unicode(true)
            .utf8(false);
        let config = dense::Config::new()
            .start_kind(StartKind::Unanchored)
            .match_kind(MatchKind::LeftmostFirst)
            .unicode_word_boundary(true)
            .minimize(false)
            .accelerate(false)
            .determinize_size_limit(Some(DETERMINIZE_SIZE_LIMIT))
            .dfa_size_limit(Some(DFA_SIZE_LIMIT));
        let dfa = dense::Builder::new()
            .configure(config)
            .syntax(syntax)
            .build(pattern)
            .map_err(|error| GpuRegexError::Compile(error.to_string()))?;

        let start = dfa
            .start_state_forward(&Input::new(b""))
            .map_err(|error| GpuRegexError::Compile(error.to_string()))?;
        if dfa.is_match_state(start) {
            return Err(GpuRegexError::EmptyPattern);
        }

        let mut compact = HashMap::new();
        let mut queue = VecDeque::new();
        compact.insert(start, 0u32);
        queue.push_back(start);
        let mut transitions = Vec::new();
        let mut equals = Vec::new();
        let mut eoi_match = Vec::new();

        while let Some(state) = queue.pop_front() {
            for &byte in BASE64.iter() {
                let next = dfa.next_state(state, byte);
                transitions.push(encode_transition(&dfa, next, &mut compact, &mut queue)?);
            }
            let next_equals = dfa.next_state(state, b'=');
            equals.push(encode_transition(
                &dfa,
                next_equals,
                &mut compact,
                &mut queue,
            )?);
            let eoi = dfa.next_eoi_state(state);
            if dfa.is_quit_state(eoi) {
                return Err(GpuRegexError::QuitStateReachable);
            }
            eoi_match.push(u8::from(dfa.is_match_state(eoi)));
            if compact.len() > GPU_STATE_LIMIT {
                return Err(GpuRegexError::TooManyStates {
                    states: compact.len(),
                    limit: GPU_STATE_LIMIT,
                });
            }
            let table_bytes = transitions.len() * std::mem::size_of::<u32>()
                + equals.len() * std::mem::size_of::<u32>()
                + eoi_match.len();
            if table_bytes > GPU_TABLE_LIMIT {
                return Err(GpuRegexError::TableTooLarge {
                    bytes: table_bytes,
                    limit: GPU_TABLE_LIMIT,
                });
            }
        }

        let state_count = u32::try_from(compact.len()).map_err(|_| GpuRegexError::InvalidState)?;
        Ok(Self {
            transitions,
            equals,
            eoi_match,
            state_count,
        })
    }

    /// Tests a Base64 slice using the same transition path used by CUDA.
    pub fn is_match_bytes(&self, haystack: &[u8]) -> Result<bool, GpuRegexError> {
        let mut state = 0usize;
        for &byte in haystack {
            let edge = if byte == b'=' {
                self.equals[state]
            } else {
                let symbol = BASE64
                    .iter()
                    .position(|&candidate| candidate == byte)
                    .ok_or(GpuRegexError::InvalidInput(byte))?;
                self.transitions[state * 64 + symbol]
            };
            if edge & DFA_MATCH != 0 {
                return Ok(true);
            }
            if edge & DFA_DEAD != 0 {
                return Ok(false);
            }
            state = (edge & DFA_STATE_MASK) as usize;
            if state >= self.state_count as usize {
                return Err(GpuRegexError::InvalidState);
            }
        }
        Ok(self.eoi_match[state] != 0)
    }

    /// Returns the compact table size in bytes.
    pub fn table_bytes(&self) -> usize {
        self.transitions.len() * std::mem::size_of::<u32>()
            + self.equals.len() * std::mem::size_of::<u32>()
            + self.eoi_match.len()
    }
}

fn encode_transition(
    dfa: &impl Automaton,
    state: regex_automata::util::primitives::StateID,
    compact: &mut HashMap<regex_automata::util::primitives::StateID, u32>,
    queue: &mut VecDeque<regex_automata::util::primitives::StateID>,
) -> Result<u32, GpuRegexError> {
    if dfa.is_match_state(state) {
        return Ok(DFA_MATCH);
    }
    if dfa.is_dead_state(state) {
        return Ok(DFA_DEAD);
    }
    if dfa.is_quit_state(state) {
        return Err(GpuRegexError::QuitStateReachable);
    }
    if let Some(&id) = compact.get(&state) {
        return Ok(id);
    }
    let id = u32::try_from(compact.len()).map_err(|_| GpuRegexError::InvalidState)?;
    if id >= GPU_STATE_LIMIT as u32 || id >= DFA_STATE_MASK {
        return Err(GpuRegexError::TooManyStates {
            states: compact.len() + 1,
            limit: GPU_STATE_LIMIT,
        });
    }
    compact.insert(state, id);
    queue.push_back(state);
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use regex::bytes::RegexBuilder;

    fn assert_equivalent(pattern: &str, inputs: &[&[u8]]) {
        let reference = RegexBuilder::new(pattern)
            .case_insensitive(false)
            .build()
            .unwrap();
        let dfa = GpuRegexDfa::compile(pattern, true).unwrap();
        for input in inputs {
            assert_eq!(
                reference.is_match(input),
                dfa.is_match_bytes(input).unwrap(),
                "pattern {pattern:?}, input {input:?}"
            );
        }
    }

    #[test]
    fn matches_anchors_and_eoi() {
        assert_equivalent("^foo", &[b"foo", b"xfoo", b"foo+", b""]);
        assert_equivalent("foo$", &[b"foo", b"xfoo", b"foobar", b""]);
        assert_equivalent(r"\Afoo\z", &[b"foo", b"xfoo", b"foo+"]);
        assert_equivalent("=+$", &[b"=", b"A=", b"==", b"A"]);
    }

    #[test]
    fn matches_classes_and_repetition() {
        assert_equivalent(
            r"(foo|bar)[0-9]{1,3}",
            &[b"foo1", b"bar123", b"bar1234", b"xxfoo9yy", b"baz1"],
        );
        assert_equivalent(r"a.{0,3}z", &[b"az", b"abz", b"abcdez", b"a123z"]);
        assert_equivalent(r"\bfoo\b", &[b"foo", b"xfoo", b"foo+", b"foobar"]);
    }

    #[test]
    fn random_base64_differential() {
        let patterns = [
            "dave",
            "d.ve",
            "[A-Za-z0-9+/]{4}",
            "^(foo|bar)",
            "a.*z",
            "(?i:wire|wg)[0-9]{2}",
        ];
        let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut seed = 0x1234_5678u32;
        for pattern in patterns {
            let reference = RegexBuilder::new(pattern)
                .case_insensitive(true)
                .build()
                .unwrap();
            let dfa = GpuRegexDfa::compile(pattern, false).unwrap();
            for _ in 0..170_000 {
                let mut input = [0u8; 44];
                for byte in &mut input {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    *byte = alphabet[(seed as usize) % alphabet.len()];
                }
                assert_eq!(
                    reference.is_match(&input),
                    dfa.is_match_bytes(&input).unwrap(),
                    "pattern {pattern:?}"
                );
            }
        }
    }

    #[test]
    fn rejects_empty_and_oversized_patterns() {
        assert!(matches!(
            GpuRegexDfa::compile("", true),
            Err(GpuRegexError::EmptyPattern)
        ));
        let oversized = "a".repeat(PATTERN_MAX_BYTES + 1);
        assert!(matches!(
            GpuRegexDfa::compile(&oversized, true),
            Err(GpuRegexError::PatternTooLong)
        ));
    }
}
