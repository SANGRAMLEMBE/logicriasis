FROM python:3.11-slim

WORKDIR /app

# System deps (git needed for unsloth install from source)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pre-install PyTorch CUDA 12.4 wheels — bundled CUDA libs work on HF GPU Spaces
# This makes train_on_hf.py skip its 30-min runtime install step entirely
RUN pip install --no-cache-dir \
    "torch==2.5.1" torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Training deps — installed at image build time, not at container start
RUN pip install --no-cache-dir \
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    "trl>=0.9.0" \
    datasets \
    accelerate \
    peft \
    bitsandbytes \
    matplotlib \
    huggingface_hub \
    requests

# API deps
RUN pip install --no-cache-dir \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.29.0" \
    "pydantic>=2.0.0" \
    "openai>=1.30.0" \
    "gradio>=6.0.0" \
    "numpy>=1.26.0" \
    "httpx>=0.27.0"

COPY . /app

EXPOSE 7860

# Longer start-period: uvicorn starts in 2s, but give 60s headroom for any import delay
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:7860/ || exit 1

# run.py starts uvicorn (health check) + spawns train_on_hf.py as background process
CMD ["python", "run.py"]
