use crate::{GpuPattern, PatternKind, SearchPattern};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use cudarc::driver::{CudaContext, CudaSlice, DriverError, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;
use x25519_dalek::StaticSecret;

const PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vanity_x25519.ptx"));
const BLOCK_SIZE: u32 = 256;

/// A reusable CUDA context and set of device buffers for vanity-key batches.
pub struct GpuSearcher {
    _context: Arc<CudaContext>,
    stream: Arc<cudarc::driver::CudaStream>,
    literal_function: cudarc::driver::CudaFunction,
    regex_function: cudarc::driver::CudaFunction,
    regex_test_function: cudarc::driver::CudaFunction,
    d_seed: CudaSlice<u8>,
    d_prefix: CudaSlice<u8>,
    d_found: CudaSlice<i32>,
    d_private: CudaSlice<u8>,
    d_public: CudaSlice<u8>,
    prepared: Option<PreparedGpuPattern>,
}

enum PreparedGpuPattern {
    Text {
        bytes: CudaSlice<u8>,
        len: u32,
        mode: u32,
        case_sensitive: u32,
    },
    Regex {
        transitions: CudaSlice<u32>,
        equals: CudaSlice<u32>,
        eoi_match: CudaSlice<u8>,
    },
}

#[derive(Debug)]
/// The outcome of one GPU kernel launch.
pub struct BatchResult {
    /// Number of candidates assigned to the launch.
    pub attempts: u64,
    /// First matching `(private_key, public_key)` pair, Base64 encoded.
    pub candidate: Option<(String, String)>,
}

impl GpuSearcher {
    /// Creates a searcher on CUDA device 0.
    pub fn new() -> Result<Self, DriverError> {
        Self::new_on_device(0)
    }

    /// Returns the number of CUDA devices visible to the process.
    pub fn device_count() -> Result<usize, DriverError> {
        Ok(CudaContext::device_count()?.max(0) as usize)
    }

    /// Creates a searcher on the visible CUDA device at `device`.
    pub fn new_on_device(device: usize) -> Result<Self, DriverError> {
        Self::new_on_device_with_optional_pattern(device, None)
    }

    /// Creates a searcher and uploads its pattern once to the selected GPU.
    pub fn new_on_device_with_pattern(
        device: usize,
        pattern: &GpuPattern,
    ) -> Result<Self, DriverError> {
        Self::new_on_device_with_optional_pattern(device, Some(pattern))
    }

    fn new_on_device_with_optional_pattern(
        device: usize,
        pattern: Option<&GpuPattern>,
    ) -> Result<Self, DriverError> {
        let context = CudaContext::new(device)?;
        let stream = context.default_stream();
        let ptx = Ptx::from_src(std::str::from_utf8(PTX).expect("nvcc emitted non-UTF-8 PTX"));
        let module = context.load_module(ptx)?;
        let literal_function = module.load_function("vanity_kernel")?;
        let regex_function = module.load_function("vanity_regex_kernel")?;
        let regex_test_function = module.load_function("regex_match_test_kernel")?;
        let d_seed = stream.clone_htod(&[0u8; 32])?;
        let d_prefix = stream.clone_htod(&[0u8; 44])?;
        let d_found = stream.clone_htod(&[-1i32])?;
        let d_private = stream.clone_htod(&[0u8; 32])?;
        let d_public = stream.clone_htod(&[0u8; 32])?;
        let prepared = match pattern {
            Some(GpuPattern::Literal {
                bytes,
                case_sensitive,
            }) => Some(PreparedGpuPattern::Text {
                bytes: stream.clone_htod(bytes)?,
                len: bytes.len() as u32,
                mode: 0,
                case_sensitive: u32::from(*case_sensitive),
            }),
            Some(GpuPattern::Glob {
                bytes,
                case_sensitive,
            }) => Some(PreparedGpuPattern::Text {
                bytes: stream.clone_htod(bytes)?,
                len: bytes.len() as u32,
                mode: 1,
                case_sensitive: u32::from(*case_sensitive),
            }),
            Some(GpuPattern::Regex(dfa)) => Some(PreparedGpuPattern::Regex {
                transitions: stream.clone_htod(&dfa.transitions)?,
                equals: stream.clone_htod(&dfa.equals)?,
                eoi_match: stream.clone_htod(&dfa.eoi_match)?,
            }),
            None => None,
        };
        Ok(Self {
            _context: context,
            stream,
            literal_function,
            regex_function,
            regex_test_function,
            d_seed,
            d_prefix,
            d_found,
            d_private,
            d_public,
            prepared,
        })
    }

    /// Searches one GPU batch for a case-insensitive literal match.
    ///
    /// The match may begin at any offset whose complete prefix lies within
    /// `start..end`. `base_counter` identifies this batch within a larger
    /// search. Invalid ranges, empty prefixes, and unsupported batch sizes
    /// return an empty result with zero attempts.
    pub fn search_batch(
        &mut self,
        prefix: &str,
        start: usize,
        end: usize,
        batch: u64,
        base_counter: u64,
    ) -> Result<BatchResult, DriverError> {
        let pattern = SearchPattern::new(prefix, PatternKind::Literal, false)
            .expect("literal patterns are always valid");
        let gpu_pattern = pattern
            .gpu_pattern()
            .expect("literal patterns support CUDA");
        self.search_batch_with_pattern(&gpu_pattern, start, end, batch, base_counter)
    }

    /// Searches one GPU batch with a literal, glob, or prepared regex pattern.
    pub fn search_batch_with_pattern(
        &mut self,
        pattern: &GpuPattern,
        start: usize,
        end: usize,
        batch: u64,
        base_counter: u64,
    ) -> Result<BatchResult, DriverError> {
        if let GpuPattern::Regex(dfa) = pattern {
            self.prepared = Some(PreparedGpuPattern::Regex {
                transitions: self.stream.clone_htod(&dfa.transitions)?,
                equals: self.stream.clone_htod(&dfa.equals)?,
                eoi_match: self.stream.clone_htod(&dfa.eoi_match)?,
            });
            return self.search_batch_prepared(start, end, batch, base_counter);
        }
        let (bytes, mode, case_sensitive) = match pattern {
            GpuPattern::Literal {
                bytes,
                case_sensitive,
            } => (bytes.as_slice(), 0, u32::from(*case_sensitive)),
            GpuPattern::Glob {
                bytes,
                case_sensitive,
            } => (bytes.as_slice(), 1, u32::from(*case_sensitive)),
            GpuPattern::Regex(_) => unreachable!("regex is handled above"),
        };
        if bytes.is_empty() || bytes.len() > 44 || end > 44 || start > end {
            return Ok(BatchResult {
                attempts: 0,
                candidate: None,
            });
        }
        let blocks = batch.div_ceil(BLOCK_SIZE as u64);
        if batch == 0 || blocks > u32::MAX as u64 {
            return Ok(BatchResult {
                attempts: 0,
                candidate: None,
            });
        }

        let seed = StaticSecret::random().to_bytes();
        let mut prefix_host = [0u8; 44];
        prefix_host[..bytes.len()].copy_from_slice(bytes);
        self.stream.memcpy_htod(&seed, &mut self.d_seed)?;
        self.stream.memcpy_htod(&prefix_host, &mut self.d_prefix)?;
        self.stream.memcpy_htod(&[-1i32], &mut self.d_found)?;

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        let prefix_len = bytes.len() as u32;
        let pattern_mode = mode;
        let start = start as u32;
        let end = end as u32;
        let mut args = self.stream.launch_builder(&self.literal_function);
        args.arg(&self.d_seed)
            .arg(&base_counter)
            .arg(&batch)
            .arg(&self.d_prefix)
            .arg(&prefix_len)
            .arg(&pattern_mode)
            .arg(&case_sensitive)
            .arg(&start)
            .arg(&end)
            .arg(&mut self.d_found)
            .arg(&mut self.d_private)
            .arg(&mut self.d_public);
        unsafe { args.launch(cfg)? };
        self.stream.synchronize()?;

        let found = self.stream.clone_dtoh(&self.d_found)?[0];
        let candidate = if found >= 0 {
            let private = self.stream.clone_dtoh(&self.d_private)?;
            let public = self.stream.clone_dtoh(&self.d_public)?;
            Some((STANDARD.encode(private), STANDARD.encode(public)))
        } else {
            None
        };
        Ok(BatchResult {
            attempts: batch,
            candidate,
        })
    }

    /// Searches a batch using a pattern uploaded by
    /// [`GpuSearcher::new_on_device_with_pattern`].
    pub fn search_batch_prepared(
        &mut self,
        start: usize,
        end: usize,
        batch: u64,
        base_counter: u64,
    ) -> Result<BatchResult, DriverError> {
        if end > 44 || start > end {
            return Ok(BatchResult {
                attempts: 0,
                candidate: None,
            });
        }
        let blocks = batch.div_ceil(BLOCK_SIZE as u64);
        if batch == 0 || blocks > u32::MAX as u64 {
            return Ok(BatchResult {
                attempts: 0,
                candidate: None,
            });
        }

        let seed = StaticSecret::random().to_bytes();
        self.stream.memcpy_htod(&seed, &mut self.d_seed)?;
        self.stream.memcpy_htod(&[-1i32], &mut self.d_found)?;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        let start = start as u32;
        let end = end as u32;
        let mut args = match self.prepared.as_ref() {
            Some(PreparedGpuPattern::Text {
                bytes,
                len,
                mode,
                case_sensitive,
            }) => {
                let mut args = self.stream.launch_builder(&self.literal_function);
                args.arg(&self.d_seed)
                    .arg(&base_counter)
                    .arg(&batch)
                    .arg(bytes)
                    .arg(len)
                    .arg(mode)
                    .arg(case_sensitive)
                    .arg(&start)
                    .arg(&end)
                    .arg(&mut self.d_found)
                    .arg(&mut self.d_private)
                    .arg(&mut self.d_public);
                args
            }
            Some(PreparedGpuPattern::Regex {
                transitions,
                equals,
                eoi_match,
            }) => {
                let mut args = self.stream.launch_builder(&self.regex_function);
                args.arg(&self.d_seed)
                    .arg(&base_counter)
                    .arg(&batch)
                    .arg(transitions)
                    .arg(equals)
                    .arg(eoi_match)
                    .arg(&start)
                    .arg(&end)
                    .arg(&mut self.d_found)
                    .arg(&mut self.d_private)
                    .arg(&mut self.d_public);
                args
            }
            None => {
                return Ok(BatchResult {
                    attempts: 0,
                    candidate: None,
                });
            }
        };
        unsafe { args.launch(cfg)? };
        self.stream.synchronize()?;

        let found = self.stream.clone_dtoh(&self.d_found)?[0];
        let candidate = if found >= 0 {
            let private = self.stream.clone_dtoh(&self.d_private)?;
            let public = self.stream.clone_dtoh(&self.d_public)?;
            Some((STANDARD.encode(private), STANDARD.encode(public)))
        } else {
            None
        };
        Ok(BatchResult {
            attempts: batch,
            candidate,
        })
    }

    /// Runs the standalone CUDA regex matcher against fixed-width test inputs.
    ///
    /// This is intended for differential tests. `inputs` must contain complete
    /// 44-byte Base64 public-key encodings (including optional `=` padding).
    pub fn test_regex_match(
        &self,
        inputs: &[[u8; 44]],
        start: usize,
        end: usize,
    ) -> Result<Vec<bool>, DriverError> {
        if inputs.is_empty() || end > 44 || start > end {
            return Ok(Vec::new());
        }
        let Some(PreparedGpuPattern::Regex {
            transitions,
            equals,
            eoi_match,
        }) = self.prepared.as_ref()
        else {
            return Ok(Vec::new());
        };
        let flat: Vec<u8> = inputs.iter().flatten().copied().collect();
        let d_inputs = self.stream.clone_htod(&flat)?;
        let count = inputs.len() as u32;
        let mut results = vec![0u8; inputs.len()];
        let mut d_results = self.stream.clone_htod(&results)?;
        let mut args = self.stream.launch_builder(&self.regex_test_function);
        let start = start as u32;
        let end = end as u32;
        args.arg(&d_inputs)
            .arg(&44u32)
            .arg(&count)
            .arg(transitions)
            .arg(equals)
            .arg(eoi_match)
            .arg(&start)
            .arg(&end)
            .arg(&mut d_results);
        let cfg = LaunchConfig {
            grid_dim: (count.div_ceil(BLOCK_SIZE), 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { args.launch(cfg)? };
        self.stream.synchronize()?;
        results = self.stream.clone_dtoh(&d_results)?;
        Ok(results.into_iter().map(|result| result != 0).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use x25519_dalek::PublicKey;

    #[test]
    #[ignore = "requires a CUDA device"]
    fn gpu_candidate_matches_dalek() {
        let mut gpu = GpuSearcher::new().expect("CUDA device is required");
        let result = gpu
            .search_batch("a", 0, 10, 1024, 0)
            .expect("CUDA kernel launch");
        let (private, public) = result.candidate.expect("one-character match");
        let private_bytes: [u8; 32] = STANDARD.decode(private).unwrap().try_into().unwrap();
        let public_bytes: [u8; 32] = STANDARD.decode(public).unwrap().try_into().unwrap();
        let expected = PublicKey::from(&StaticSecret::from(private_bytes));
        assert_eq!(expected.to_bytes(), public_bytes);
    }
}
