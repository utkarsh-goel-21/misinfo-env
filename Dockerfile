# ═══════════════════════════════════════
# SENTINEL-9 — Multi-stage Docker Build
# Optimized for HuggingFace Spaces (2 vCPU, 8GB RAM)
# ═══════════════════════════════════════

FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ═══════════════════════════════════════
FROM python:3.11-slim

LABEL maintainer="sentinel-team"
LABEL version="2.0.0"
LABEL description="SENTINEL-9 Misinformation Containment — OpenEnv POMDP Benchmark"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project
COPY . .

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860
ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV API_KEY=""
ENV HF_TOKEN=""

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
