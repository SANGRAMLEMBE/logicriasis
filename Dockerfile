FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.29.0" \
    "pydantic>=2.0.0" \
    "openai>=1.30.0" \
    "numpy>=1.26.0" \
    "httpx>=0.27.0" \
    "huggingface_hub" \
    "requests"

COPY . /app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "run.py"]
