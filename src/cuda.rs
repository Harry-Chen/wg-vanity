use base64::{Engine as _, engine::general_purpose::STANDARD};
use cudarc::driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;
use x25519_dalek::StaticSecret;

const PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vanity_x25519.ptx"));
const BLOCK_SIZE: u32 = 256;

pub struct GpuSearcher {
    _context: Arc<CudaContext>,
    stream: Arc<cudarc::driver::CudaStream>,
    function: cudarc::driver::CudaFunction,
}

#[derive(Debug)]
pub struct BatchResult {
    pub attempts: u64,
    pub candidate: Option<(String, String)>,
}

impl GpuSearcher {
    pub fn new() -> Result<Self, DriverError> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let ptx = Ptx::from_src(std::str::from_utf8(PTX).expect("nvcc emitted non-UTF-8 PTX"));
        let module = context.load_module(ptx)?;
        let function = module.load_function("vanity_kernel")?;
        Ok(Self {
            _context: context,
            stream,
            function,
        })
    }

    pub fn search_batch(
        &self,
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
        let d_seed = self.stream.clone_htod(&seed)?;
        let d_prefix = self.stream.clone_htod(prefix_bytes)?;
        let mut d_attempts = self.stream.clone_htod(&[0u64])?;
        let mut d_found = self.stream.clone_htod(&[-1i32])?;
        let mut d_private = self.stream.clone_htod(&[0u8; 32])?;
        let mut d_public = self.stream.clone_htod(&[0u8; 32])?;

        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        let prefix_len = prefix_bytes.len() as u32;
        let start = start as u32;
        let end = end as u32;
        let mut args = self.stream.launch_builder(&self.function);
        args.arg(&d_seed)
            .arg(&base_counter)
            .arg(&batch)
            .arg(&d_prefix)
            .arg(&prefix_len)
            .arg(&start)
            .arg(&end)
            .arg(&mut d_attempts)
            .arg(&mut d_found)
            .arg(&mut d_private)
            .arg(&mut d_public);
        unsafe { args.launch(cfg)? };
        self.stream.synchronize()?;

        let attempts = self.stream.clone_dtoh(&d_attempts)?[0];
        let found = self.stream.clone_dtoh(&d_found)?[0];
        let candidate = if found >= 0 {
            let private = self.stream.clone_dtoh(&d_private)?;
            let public = self.stream.clone_dtoh(&d_public)?;
            Some((STANDARD.encode(private), STANDARD.encode(public)))
        } else {
            None
        };
        Ok(BatchResult {
            attempts,
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
        let gpu = GpuSearcher::new().expect("CUDA device is required");
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
