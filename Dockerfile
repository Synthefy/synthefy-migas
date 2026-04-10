# ── Stage 1: builder ───────────────────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN uv sync --frozen --no-dev --extra api

# ── Stage 2: runtime ──────────────────────────────────────────
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv libpython3.12 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Fix venv symlink: builder has python at /usr/local/bin/python3,
# but runtime (deadsnakes) has it at /usr/bin/python3.12
RUN ln -sf /usr/bin/python3.12 /app/.venv/bin/python

ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"
ENV HF_HOME="/app/.cache/huggingface"

# SageMaker sends "serve" as the command — create a script for it.
# SageMaker uses port 8080 by default.
RUN printf '#!/bin/bash\nexec python -m uvicorn migaseval.api.main:app --host 0.0.0.0 --port 8080\n' > /usr/local/bin/serve && \
    chmod +x /usr/local/bin/serve

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["serve"]
