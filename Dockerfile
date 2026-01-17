FROM nvcr.io/nvidia/pytorch:25.12-py3

# Declare ARGs to receive values from docker-compose
ARG GPU_ARCH
ARG MAX_JOBS

# Metadata for sm_120 (Blackwell) hardware alignment
ENV VLLM_GPU_ARCH=${GPU_ARCH}
ENV TORCH_CUDA_ARCH_LIST=${GPU_ARCH}
ENV NCCL_DMABUF_ENABLE=1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y build-essential cmake ninja-build && rm -rf /var/lib/apt/lists/*

WORKDIR /vllm-workspace

# Optimization: Pre-installing standard heavy dependencies to leverage Docker layer caching
RUN pip install --upgrade pip && \
    pip install uvloop sentencepiece \
                fastapi uvicorn pydantic prometheus-client \
                lm-format-enforcer outlines kernels

# Context: Copying local vLLM source for custom sm_120 kernel compilation
COPY vllm-src /vllm-workspace

# Performance: Threadripper-optimized parallel compilation. 
# Logic: Dynamic job scaling to prevent OOM while maximizing 128GB RAM throughput.
RUN ACTUAL_JOBS=${MAX_JOBS}; \
    if [ "$ACTUAL_JOBS" -eq "4" ]; then \
        CPUS=$(nproc); \
        ACTUAL_JOBS=$(( CPUS > 8 ? 8 : CPUS )); \
    fi; \
    echo "Compiling with $ACTUAL_JOBS threads for SM_120 target..." && \
    MAX_JOBS=$ACTUAL_JOBS pip install -e .

# Critical Fix: Removal of legacy flash-attn to force the engine to use FlashInfer backend,
# avoiding 'undefined symbol' errors prevalent in Blackwell's new instruction set.
RUN pip uninstall -y flash-attn

EXPOSE 8000
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
