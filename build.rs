use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/vanity_x25519.cu");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let cuda_home = env::var_os("CUDA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    let nvcc = cuda_home.join("bin/nvcc");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let ptx = out_dir.join("vanity_x25519.ptx");

    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "compute_120".to_string());
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
