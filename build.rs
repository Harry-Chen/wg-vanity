use std::env;
use std::path::PathBuf;
use std::process::Command;

const FALLBACK_ARCH: &str = "compute_80";

fn detect_cuda_arch() -> String {
    if let Ok(arch) = env::var("CUDA_ARCH") {
        return arch;
    }

    let detected = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| {
            let capabilities = String::from_utf8(output.stdout).ok()?;
            capabilities
                .lines()
                .filter_map(|line| {
                    let (major, minor) = line.trim().split_once('.')?;
                    Some((major.parse::<u32>().ok()?, minor.parse::<u32>().ok()?))
                })
                .min()
        });

    if let Some((major, minor)) = detected {
        let arch = format!("compute_{major}{minor}");
        println!("cargo:warning=detected CUDA architecture {arch}");
        arch
    } else {
        println!("cargo:warning=CUDA architecture not detected; using {FALLBACK_ARCH} PTX");
        FALLBACK_ARCH.to_string()
    }
}

fn main() {
    println!("cargo:rerun-if-changed=cuda/vanity_x25519.cu");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_VISIBLE_DEVICES");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let cuda_home = env::var_os("CUDA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    let nvcc = cuda_home.join("bin/nvcc");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let ptx = out_dir.join("vanity_x25519.ptx");

    let arch = detect_cuda_arch();
    let mut command = Command::new(&nvcc);
    let status = command
        .args(["-O3", "--std=c++17"])
        .arg(format!("-arch={arch}"))
        .args(["-ptx", "cuda/vanity_x25519.cu", "-o"])
        .arg(&ptx)
        .status()
        .unwrap_or_else(|err| panic!("failed to execute {}: {err}", nvcc.display()));
    assert!(
        status.success(),
        "nvcc failed while compiling {}",
        ptx.display()
    );
}
