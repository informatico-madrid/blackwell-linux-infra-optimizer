FROM nvcr.io/nvidia/pytorch:25.12-py3

ENV VLLM_GPU_ARCH=12.0
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV NCCL_DMABUF_ENABLE=1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y build-essential cmake ninja-build && rm -rf /var/lib/apt/lists/*

WORKDIR /vllm-workspace

# CAPA DE LIBRERÍAS (Sin bitsandbytes y sin flash-attn)
RUN pip install --upgrade pip && \
    pip install uvloop sentencepiece \
                fastapi uvicorn pydantic prometheus-client \
                lm-format-enforcer outlines kernels

# CAPA DE COMPILACIÓN (vLLM puro)
COPY vllm-src /vllm-workspace
RUN MAX_JOBS=10 pip install -e .

# LIMPIEZA POST-INSTALACIÓN (Aseguramos que flash-attn no exista)
RUN pip uninstall -y flash-attn

EXPOSE 8000
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
