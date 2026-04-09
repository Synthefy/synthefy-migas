# Deploying Migas-1.5

Three deployment options:

| Option | Best for | Summary generation |
|--------|----------|-------------------|
| **Baseten (Truss)** | Managed cloud, auto-scaling, no infra to manage | Via Baseten Model APIs |
| **Docker** | Self-hosting on any cloud or on-prem | Via vLLM sidecar or bring your own |
| **Docker (CPU)** | Testing without GPU | Pre-computed summaries only |

---

## Install

```bash
# Inference only (base)
uv sync

# Inference + FastAPI server (Docker)
uv sync --extra api

# Inference + evaluation baselines (notebooks, metrics)
uv sync --extra eval

# Everything
uv sync --extra api --extra eval
```

Base dependencies are inference-only: torch, chronos-forecasting, sentence-transformers, timesfm, openai, pandas, numpy. Evaluation deps (tabpfn, prophet, pmdarima, yfinance, polars, anthropic, etc.) are in the `eval` extra and are **not** installed in Docker or Baseten.

---

## Option 1: Baseten (recommended for production)

Baseten manages GPU provisioning, scaling, and serving. You deploy via the Truss CLI.

### Setup

```bash
pip install truss
truss login  # paste your Baseten API key
```

### Set secrets

In the Baseten dashboard under **Settings → Secrets**, add:

| Secret | Value |
|--------|-------|
| `baseten_api_key` | Your Baseten API key (for calling managed LLM endpoints) |

### Deploy

```bash
cd truss
truss push
```

Baseten builds the container, provisions a T4 GPU, downloads model weights, and gives you an endpoint.

### Test

**Mode 1 — text in, LLM generates summaries:**
```bash
curl -X POST https://model-XXXX.api.baseten.co/environments/production/predict \
  -H "Authorization: Api-Key YOUR_BASETEN_KEY" \
  -H "Content-Type: application/json" \
  -d @truss/test_mode1.json
```

**Mode 2 — pre-computed summaries:**
```bash
curl -X POST https://model-XXXX.api.baseten.co/environments/production/predict \
  -H "Authorization: Api-Key YOUR_BASETEN_KEY" \
  -H "Content-Type: application/json" \
  -d @truss/test_mode2.json
```

Summary generation uses Baseten's managed LLM (`openai/gpt-oss-120b` by default) — no separate vLLM deployment needed.

### Cost

- Migas on T4: ~$0.63/hr (scales to zero when idle)
- LLM calls: per-token pricing via Baseten Model APIs

---

## Option 2: Docker (self-hosted)

### Migas only (no vLLM)

You provide pre-computed summaries or point to your own LLM server.

```bash
docker compose up -d
```

### Migas + vLLM (self-contained)

Two containers: Migas on one GPU, vLLM on another.

```bash
docker compose -f docker-compose.yml -f docker-compose.vllm.yml up -d
```

### CPU only

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace token (if model repo is private) |
| `MIGAS_API_KEY` | — | If set, requests must include `X-API-Key` header |
| `MIGAS_DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |
| `MIGAS_MODEL` | `Synthefy/migas-1.5` | HuggingFace repo ID or local path |
| `MIGAS_GPU` | `0` | GPU index for Migas container |
| `VLLM_MODEL` | `openai/gpt-oss-120b` | LLM model for summary generation |
| `VLLM_GPUS` | `1` | GPU index(es) for vLLM container |
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for vLLM |
| `VLLM_MAX_MODEL_LEN` | `32768` | Max context length |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.60` | Fraction of GPU VRAM for vLLM |

### GPU assignment

By default, Migas uses GPU 0 and vLLM uses GPU 1. Override with:

```bash
# 4 GPUs: Migas on GPU 0, vLLM sharded across GPUs 1,2,3
MIGAS_GPU=0 VLLM_GPUS=1,2,3 VLLM_TENSOR_PARALLEL_SIZE=3 \
  docker compose -f docker-compose.yml -f docker-compose.vllm.yml up -d

# Single GPU (shared): both on GPU 0, limit vLLM memory
MIGAS_GPU=0 VLLM_GPUS=0 VLLM_GPU_MEMORY_UTILIZATION=0.40 \
  docker compose -f docker-compose.yml -f docker-compose.vllm.yml up -d
```

### Choosing a vLLM model

| Model | VRAM needed | Quality | Setup |
|-------|------------|---------|-------|
| `openai/gpt-oss-120b` | ~80 GB (2-4x A100) | Best | `VLLM_TENSOR_PARALLEL_SIZE=2` |
| `Qwen/Qwen3-8B` | ~16 GB (1x A10G) | Good | `VLLM_MODEL=Qwen/Qwen3-8B` |
| `Qwen/Qwen3-4B` | ~8 GB (1x T4) | Acceptable | `VLLM_MODEL=Qwen/Qwen3-4B` |
| `microsoft/phi-4` | ~8 GB (1x T4) | Good | `VLLM_MODEL=microsoft/phi-4` |

### GitHub Container Registry

Push to `main` or tag a release — the GitHub Actions workflow builds and pushes automatically:

```bash
docker pull ghcr.io/synthefy/migas-1.5:latest
docker run -d -p 8080:8080 --gpus 1 ghcr.io/synthefy/migas-1.5:latest
```

---

## GPU sizing guide

### Migas only

Migas-1.5 uses ~2 GB VRAM (Chronos-2 + FinBERT + fusion head, all float32).

| GPU | VRAM | Cost | Verdict |
|-----|------|------|---------|
| T4 | 16 GB | ~$0.63/hr | Best fit — 8x headroom |
| L4 | 24 GB | ~$0.85/hr | Overkill |
| A10G | 24 GB | ~$1.21/hr | Overkill |

### Migas + vLLM

| Setup | Migas GPU | vLLM GPU | vLLM model | Total cost |
|-------|-----------|----------|------------|------------|
| Budget | 1x T4 | 1x T4 | Qwen3-4B | ~$1.26/hr |
| Balanced | 1x T4 | 1x A10G | Qwen3-8B | ~$1.84/hr |
| Best quality | 1x T4 | 2x A100 | gpt-oss-120b | ~$8.63/hr |

---

## API reference

### `GET /health`

```json
{"status": "ok", "model_loaded": true, "device": "cuda", "version": "1.5.0"}
```

### `POST /predict`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `dates` | `list[str]` | Yes | — | Date strings (YYYY-MM-DD) |
| `values` | `list[float]` | Yes | — | Time series values |
| `text` | `list[str]` | Mode 1 | — | Per-timestep text. Triggers LLM summary generation |
| `summaries` | `list[str] \| str` | Mode 2 | — | Pre-computed summary. Skips LLM |
| `series_name` | `str` | No | `"series"` | Name used in summary generation prompt |
| `pred_len` | `int` | No | `16` | Forecast horizon (1-16) |
| `seq_len` | `int` | No | all | Use last N rows only |
| `n_summaries` | `int` | No | `5` | Ensemble size (Mode 1 only) |
| `return_univariate` | `bool` | No | `false` | Also return Chronos-2 baseline |

Either `text` or `summaries` must be provided. If both are given, `summaries` takes priority.

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `forecast` | `list[float]` | Predicted values |
| `dates` | `list[str]` | Forecast dates (business days) |
| `summaries` | `list[str] \| null` | Generated summaries (Mode 1 only) |
| `univariate_forecast` | `list[float] \| null` | Chronos-2 baseline (if requested) |

### `GET /`

```json
{"name": "Migas-1.5", "version": "1.5.0", "vllm_url": "...", "vllm_model": "..."}
```

---

## Authentication

### Docker

Set `MIGAS_API_KEY` to require an API key:

```bash
MIGAS_API_KEY=my-secret-key docker compose up -d

curl -X POST http://localhost:8080/predict \
  -H "X-API-Key: my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```

### Baseten

Authentication is handled by Baseten — pass your API key in the `Authorization` header:

```bash
curl -X POST https://model-XXXX.api.baseten.co/environments/production/predict \
  -H "Authorization: Api-Key YOUR_BASETEN_KEY" \
  -d '{ ... }'
```

---

## Summary format

Summaries must contain exactly two sections:

```
FACTUAL SUMMARY:
[Dense analytical paragraph about observed facts, price ranges, percentage moves,
named catalysts, and structural dynamics from the historical window]

PREDICTIVE SIGNALS:
[Dense analytical paragraph about forward-looking information, analyst views,
upcoming catalysts, market dynamics — relative terms only, no absolute price targets]
```

This is the format Migas-1.5 was trained on. Deviating from it degrades forecast quality.
