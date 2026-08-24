# CUDA 后端与 Rust CUDA 调研

## 当前实现

仓库现在提供了一个可选的 `cuda` feature：Rust 负责 CUDA context、内存、kernel launch、结果编码和命令行，设备端 kernel 位于 `cuda/vanity_x25519.cu`，由 `nvcc` 在 Cargo build script 中编译成 PTX，再由 `cudarc` 加载。

设备端每个线程处理一个候选：

1. 使用主机 CSPRNG 生成 256-bit seed。
2. 设备端以 seed + 独立 counter 运行 ChaCha20，得到候选私钥。
3. 用 Montgomery ladder 计算 X25519 公钥。
4. 直接在设备端 Base64 编码并做大小写不敏感的区间匹配。
5. 通过 `atomicCAS` 只回传第一个匹配的原始私钥/公钥。

这不是把 `OsRng` 或 `x25519-dalek` 放进 GPU。设备端没有复用 CPU 库，且随机数不是普通的快速 PRNG：每个 batch 都使用新的主机 CSPRNG seed，ChaCha20 只负责高吞吐的设备内展开。

## 构建和使用

普通 CPU 版本不需要 CUDA：

```bash
cargo test
cargo run --release --bin wireguard-vanity-address -- dave
```

5090/Blackwell 版本需要 CUDA Toolkit 13.x（或能生成 `compute_120` PTX 的更新版）：

```bash
CUDA_HOME=/usr/local/cuda-13.3 \
  cargo build --release --features cuda --bin wireguard-vanity-address-cuda

CUDA_HOME=/usr/local/cuda-13.3 \
  ./target/release/wireguard-vanity-address-cuda dave --batch 1048576
```

默认 CPU/CUDA 搜索都会持续到找到匹配项或按 `Ctrl-C` 停止；使用限制参数可以做可控运行：

```bash
# CPU: 最多尝试 10M 个候选
cargo run --release --bin wireguard-vanity-address -- dave --trials 10000000

# CUDA: 最多运行 60 秒；duration 按 batch 检查，想要更细粒度就减小 --batch
CUDA_HOME=/usr/local/cuda-13.3 \
  ./target/release/wireguard-vanity-address-cuda dave \
  --duration 60 --batch 1048576

# CUDA: 次数、时间、kernel launch 次数可以同时给出，先到的限制生效
CUDA_HOME=/usr/local/cuda-13.3 \
  ./target/release/wireguard-vanity-address-cuda dave \
  --trials 10000000 --duration 60 --batches 100
```

`cuda` 是 Cargo feature，不是运行时 cfg：没有 `--features cuda` 时，`src/cuda.rs` 和 CUDA binary 不参与编译；启用后 `build.rs` 调用 `nvcc` 生成 `OUT_DIR/vanity_x25519.ptx`，再由 Rust host 通过 `cudarc` 加载。默认目标是 `compute_120`，也可以显式指定：

```bash
CUDA_HOME=/usr/local/cuda-13.3 CUDA_ARCH=compute_120 \
  cargo build --release --features cuda --bin wireguard-vanity-address-cuda
```

只做吞吐测试，不等待匹配结果：

```bash
CUDA_HOME=/usr/local/cuda-13.3 \
  ./target/release/wireguard-vanity-benchmark --backend cpu --trials 8000000

CUDA_HOME=/usr/local/cuda-13.3 \
  ./target/release/wireguard-vanity-benchmark --backend cuda \
    --trials 8000000 --batch 8000000
```

## 5090 实测

环境：RTX 5090（SM 12.0，32 GiB）、NVIDIA driver 610.57.04、CUDA Toolkit 13.3。

在 2026-08-24 的一次长批次测试中：

| 后端 | 候选数 | 时间 | 吞吐 |
| --- | ---: | ---: | ---: |
| CPU + Rayon | 16,000,000 | 3.517 s | 4.55 M keys/s |
| CUDA + Rust host | 16,000,000 | 0.337 s | 47.54 M keys/s |

这个数字包含每个 CUDA batch 的 seed/参数复制、kernel launch、同步和计数回读，但不包含首次 CUDA context 初始化。对应本次实现约 10.45x 加速。

初版 16-limb kernel 曾使用 255 registers/thread，并产生 48B spill stores/loads；profile 指向寄存器/local-memory 溢出是主要瓶颈。当前版本改为 5x51-bit field limbs、专用 squaring 和 addition-chain inversion：128 registers/thread、0 spill，Nsight Compute 报告 compute throughput 约 80.6%、DRAM throughput 约 0%。专用 squaring 相比通用乘法又带来约 15% 吞吐提升。

## Rust-native CUDA 评估

我实际下载了 NVIDIA `cuda-oxide` 并在这台机器上安装了当前 nightly、Clang 19 和 LLVM 工具链。`cargo-oxide doctor` 已通过：CUDA 13.3、RTX 5090、libNVVM、nvJitLink、LLVM/Clang 均可见。

但仓库固定的 `nightly-2026-04-03` 在当前 rustup 镜像中不存在；临时改为 nightly 1.100.0 后，`rustc-codegen-cuda` backend 在 `rustc_public` API 处出现 26 个兼容性错误（例如 `CrateDefType::ty_with_args` trait 导入和 MIR `Rvalue::Use` 新字段）。因此本轮没有把生产 kernel 迁移到 oxide，也没有把一个未能生成 PTX 的原型伪装成可用实现。

当前生态分成三类：

- `cudarc`/`cust`：Rust 的 CUDA driver/host bindings，设备代码仍由 CUDA C++、PTX 或 NVRTC 提供。当前后端使用 `cudarc`。
- `Rust-CUDA`：通过 `rustc_codegen_nvvm` 把 Rust 编译为 NVVM/PTX，文档明确要求 nightly、旧版 LLVM/NVVM 环境，且没有稳定 crates release。
- NVIDIA 的 `cuda-oxide`：实验性的 Rust-to-CUDA `rustc` backend，目标是用纯 Rust 表达 CUDA SIMT、warp、TMA 等模型；项目文档仍标为 alpha，并要求 pinned nightly、LLVM 21+ 和 CUDA 12.x+。

因此目前生产路径保持“Rust host + 可审计的 CUDA kernel”。oxide 值得继续跟踪，但要等它发布与当前 nightly 对齐的 backend，或单独维护一组 API-compatibility patch 后，再重写一个独立的 X25519 correctness kernel，与这里的 CUDA kernel 做 RFC 7748/`x25519-dalek` 交叉验证。

## 密钥安全注意事项

GPU 搜索会把私钥短暂放在 device global memory，并回传第一个匹配项。不要在不可信的机器或多租户 GPU 上运行；匹配后应尽快写入 WireGuard 配置并销毁进程。CUDA 路径适合生成新密钥，不会复用已有私钥，也不会把私钥作为命令行参数传入。
