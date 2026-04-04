# ─────────────────────────────────────────
# Base image — slim Python 3.11
# ─────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="your-name"
LABEL version="1.0.0"
LABEL description="Misinformation Containment Network — OpenEnv Environment"

# ─────────────────────────────────────────
# System dependencies
# ─────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────
# Working directory
# ─────────────────────────────────────────
WORKDIR /app

# ─────────────────────────────────────────
# Install Python dependencies first
# (separate layer for better caching)
# ─────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────
# Copy project files
# ─────────────────────────────────────────
COPY . .

# ─────────────────────────────────────────
# Environment variables
# ─────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860

# API credentials — must be passed at runtime
# docker run -e API_BASE_URL=... -e MODEL_NAME=... -e HF_TOKEN=...
ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""

# ─────────────────────────────────────────
# Expose port for HuggingFace Spaces
# ─────────────────────────────────────────
EXPOSE 7860

# ─────────────────────────────────────────
# Health check
# ─────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ─────────────────────────────────────────
# Start server
# ─────────────────────────────────────────
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]