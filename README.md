# vLLM Stack for NVIDIA Blackwell (SM_120) on Linux Kernel 6.14 🏎️

This repository provides a production-ready deployment stack optimized for **NVIDIA Blackwell (RTX 5090)**. It specifically addresses the integration challenges between the **Linux Kernel 6.14+** and the **sm_120 architecture**.

## 🧠 The Bridge: Solving Kernel-Architecture Incompatibilities

Standard LLM deployments fail on Blackwell/Kernel 6.14 due to unstable memory mapping and peer-to-peer deadlocks. This stack implements critical workarounds:

- **Kernel 6.14 DMA-BUF Integration**: Uses `NCCL_DMABUF_ENABLE=1` to leverage native kernel memory handling, replacing the unstable `nvidia_peermem` module.
- **SM_120 Hardware Alignment**: Specifically tuned for Blackwell's compute capability 12.0, fixing the "garbage character" output issue through **AWQ (4-bit)** quantization.
- **Attention Backend Pivot**: Forced removal of legacy `flash-attn` in favor of **FlashInfer**, bypassing symbol errors in the new hardware instruction set.
- **Memory Segmentation**: Optimized `PYTORCH_ALLOC_CONF` for the new kernel's memory management to prevent VRAM fragmentation.
- **Build-Time Resilience**: Hardened Git configuration during Docker build to prevent RPC/CURL failures when fetching massive dependencies like Triton Kernels.

## 📊 Performance (2x RTX 5090)
- **Throughput**: ~30.5 tokens/s (DeepSeek-R1-32B)
- **Latency**: < 200ms TTFT
- **Bus**: PCIe Gen 5 Direct Access (NCCL P2P PCI)

## 🛠️ Prerequisites & Manual Setup

Due to the size of the components and the bleeding-edge nature of the hardware, follow these steps before deploying:

### 1. Model Weights (AWQ)
Download the optimized weights to avoid SM_120 kernel corruption:
```bash
pip install huggingface-hub
huggingface-cli download casperhansen/deepseek-r1-distill-qwen-32b-awq --local-dir ./models/deepseek-r1-32b-awq
```

### 2. vLLM Source
Clone the source code manually to the `vllm-src` directory (using shallow clone to avoid network issues):
```bash
git clone --depth 1 [https://github.com/vllm-project/vllm.git](https://github.com/vllm-project/vllm.git) vllm-src
```

## 🚀 Quick Start

1. Clone the repository:
   git clone https://github.com/informatico-madrid/blackwell-linux-infra-optimizer

2. Prepare Environment:
   cp .env.example .env (Add your HF_TOKEN)

3. Launch:
   docker compose up --build -d

## 🤝 Support
This project bridges the gap for early adopters of Blackwell hardware. If this saves you hours of debugging, please give it a star! ⭐
