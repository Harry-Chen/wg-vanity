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
    function: cudarc::driver::CudaFunction,
    d_seed: CudaSlice<u8>,
    d_prefix: CudaSlice<u8>,
    d_found: CudaSlice<i32>,
    d_private: CudaSlice<u8>,
    d_public: CudaSlice<u8>,
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
        let context = CudaContext::new(device)?;
        let stream = context.default_stream();
        let ptx = Ptx::from_src(std::str::from_utf8(PTX).expect("nvcc emitted non-UTF-8 PTX"));
        let module = context.load_module(ptx)?;
        let function = module.load_function("vanity_kernel")?;
        let d_seed = stream.clone_htod(&[0u8; 32])?;
        let d_prefix = stream.clone_htod(&[0u8; 44])?;
        let d_found = stream.clone_htod(&[-1i32])?;
        let d_private = stream.clone_htod(&[0u8; 32])?;
        let d_public = stream.clone_htod(&[0u8; 32])?;
        Ok(Self {
            _context: context,
            stream,
            function,
            d_seed,
            d_prefix,
            d_found,
            d_private,
            d_public,
        })
    }

    /// Searches one GPU batch for a case-insensitive prefix match.
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
        if prefix.is_empty() || prefix.len() > 44 || end > 44 || start > end {
            return Ok(BatchResult {
                attempts: 0,
                candidate: None,
            });
        }
        let prefix = prefix.to_ascii_lowercase();
        let prefix_bytes = prefix.as_bytes();
        let blocks = batch.div_ceil(BLOCK_SIZE as u64);
        if batch == 0 || blocks > u32::MAX as u64 {
            return Ok(BatchResult {
                attempts: 0,
                candidate: None,
            });
        }

        let seed = StaticSecret::random().to_bytes();
        let mut prefix_host = [0u8; 44];
        prefix_host[..prefix_bytes.len()].copy_from_slice(prefix_bytes);
        self.stream.memcpy_htod(&seed, &mut self.d_seed)?;
        self.stream.memcpy_htod(&prefix_host, &mut self.d_prefix)?;
        self.stream.memcpy_htod(&[-1i32], &mut self.d_found)?;

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        let prefix_len = prefix_bytes.len() as u32;
        let start = start as u32;
        let end = end as u32;
        let mut args = self.stream.launch_builder(&self.function);
        args.arg(&self.d_seed)
            .arg(&base_counter)
            .arg(&batch)
            .arg(&self.d_prefix)
            .arg(&prefix_len)
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use x25519_dalek::PublicKey;

    #[test]
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
