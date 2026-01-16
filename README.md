# vLLM Optimized Stack for NVIDIA Blackwell (RTX 5090 / SM_120) 🏎️

This repository provides a production-ready, Docker-based deployment for Large Language Models (LLMs) specifically optimized for the **NVIDIA Blackwell (Architecture 12.0)** series.

Developed to stabilize **DeepSeek-R1-32B** on consumer-grade Blackwell hardware, this stack resolves critical compatibility gaps and performance bottlenecks in current vLLM/PyTorch releases.

## 🧠 Solved Technical Challenges

Deploying on the RTX 5090 (sm_120) currently faces several "day-one" issues. This configuration successfully mitigates:

* **Output Corruption Fix**: Migrated from bitsandbytes (8-bit) to **AWQ (4-bit)**. This resolves the "garbage character" output (!!!!!!) caused by incompatible quantization kernels on SM_120.
* **Flash-Attention Symbol Resolution**: Fixed the `undefined symbol` ImportError by performing a forced uninstall of `flash-attn` and pivoting to the **FlashInfer** backend, ensuring stable attention kernels.
* **CUDA Graph Stability**: Implemented `--enforce-eager` to bypass block reservation failures during the warmup phase, a common issue in early SM_120 driver implementations.
* **Kernel 6.14 & NCCL Optimization**: Enabled `expandable_segments` and `NCCL_DMABUF_ENABLE=1` to leverage the native memory subsystem of Kernel 6.14, replacing unstable `nvidia_peermem` modules.
* **P2P Communication**: Forced `NCCL_P2P_LEVEL=PCI` to guarantee reliable data transfer between GPUs via the **PCIe Gen 5** bus.

## 📊 Performance Benchmarks (2x RTX 5090)
* **Model**: DeepSeek-R1-Distill-Qwen-32B (AWQ)
* **Throughput**: ~30.5 tokens/s
* **Latency (TTFT)**: < 200ms
* **VRAM Efficiency**: ~10GB per GPU for weights, leaving ~22GB for massive KV Cache.

## 🚀 Quick Start

1. **Clone the repository**:
   git clone <your-repo-link>

2. **Configure Secrets**:
   cp .env.example .env (Then add your HF_TOKEN)

3. **Deploy with Docker Compose**:
   docker compose up --build -d

## 🤝 Community & Support
This stack was built to save time for researchers and developers facing the initial hurdles of the Blackwell architecture. If you find this useful, feel free to contribute or report issues.
