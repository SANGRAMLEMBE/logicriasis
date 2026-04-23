FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (inference + API only — no torch for HF Spaces CPU tier)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    "uvicorn[standard]>=0.29.0" \
    "pydantic>=2.0.0" \
    "openai>=1.30.0" \
    "gradio>=6.0.0" \
    "numpy>=1.26.0" \
    "httpx>=0.27.0"

# Copy project
COPY . .

# HF Spaces expects port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run FastAPI server — required for HF Spaces ping (GET /) and reset() (POST /reset)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
