# Deploying Migas-1.5

Four deployment options:

| Option | Best for | Summary generation |
|--------|----------|-------------------|
| **SageMaker + Bedrock** | AWS production, managed scaling | AWS Bedrock (Claude, Llama, Mistral, etc.) |
| **Baseten (Truss)** | Managed cloud, auto-scaling, no infra to manage | Via Baseten Model APIs |
| **Docker + vLLM** | Self-hosting with self-hosted LLM | vLLM sidecar (any HF model) |
| **Docker (standalone)** | Self-hosting with external LLM or pre-computed summaries | Bedrock, OpenAI, Anthropic API, or none |

---

## Install

```bash
# Inference only (base)
uv sync

# Inference + FastAPI server (Docker / SageMaker)
uv sync --extra api

# Inference + evaluation baselines (notebooks, metrics)
uv sync --extra eval

# Everything
uv sync --extra api --extra eval
```

Base dependencies are inference-only: torch, chronos-forecasting, sentence-transformers, timesfm, openai, pandas, numpy. Evaluation deps (tabpfn, prophet, pmdarima, yfinance, polars, anthropic, etc.) are in the `eval` extra and are **not** installed in Docker or Baseten.

---

## Option 1: SageMaker + Bedrock (recommended for AWS)

Migas runs as a SageMaker endpoint. Summary generation uses AWS Bedrock — no self-hosted LLM needed.

### Build and push to ECR

```bash
# Build the image
docker build -t migas-1.5 .

# Tag and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker tag migas-1.5 ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/migas-1.5:latest
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/migas-1.5:latest
```

### SageMaker endpoint configuration

The container exposes SageMaker-compatible routes:

- `GET /ping` — health check
- `POST /invocations` — inference

Set these environment variables on the SageMaker model:

| Variable | Value |
|----------|-------|
| `LLM_PROVIDER` | `bedrock` |
| `LLM_MODEL` | `anthropic.claude-3-5-haiku-20241022-v1:0` (or any Bedrock model) |
| `AWS_DEFAULT_REGION` | `us-east-1` |

Bedrock authentication uses the SageMaker execution role — no API key needed. Ensure the role has `bedrock:InvokeModel` permission.

### Available Bedrock models

| Model | Bedrock ID |
|-------|-----------|
| Claude 3.5 Haiku | `anthropic.claude-3-5-haiku-20241022-v1:0` |
| Claude 3.7 Sonnet | `anthropic.claude-3-7-sonnet-20250219-v1:0` |
| Llama 3.3 70B | `meta.llama3-3-70b-instruct-v1:0` |
| Mistral Large | `mistral.mistral-large-2407-v1:0` |

Full list: `aws bedrock list-foundation-models --query "modelSummaries[].modelId"`

### Test

**Mode 1 — text in, Bedrock generates summaries:**
```bash
curl -X POST https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/YOUR_ENDPOINT/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "series_name": "US_gasoline",
    "dates": ["2020-03-01", "2020-03-08", "2020-03-15"],
    "values": [2.555, 2.514, 2.468],
    "text": ["OPEC talks stall", "Oil price war begins", "COVID lockdowns expand"],
    "n_summaries": 3,
    "pred_len": 16
  }'
```

**Mode 2 — pre-computed summaries (no LLM needed):**
```bash
curl -X POST https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/YOUR_ENDPOINT/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dates": ["2020-03-01", "2020-03-08", "2020-03-15"],
    "values": [2.555, 2.514, 2.468],
    "summaries": ["FACTUAL SUMMARY: ... \n\nPREDICTIVE SIGNALS: ..."],
    "pred_len": 16
  }'
```

---

## Option 2: Baseten (Truss)

Baseten manages GPU provisioning, scaling, and serving. You deploy via the Truss CLI.

### Setup

```bash
pip install truss
truss login  # paste your Baseten API key
```

### Set secrets

In the Baseten dashboard under **Settings > Secrets**, add:

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

```bash
curl -X POST https://model-XXXX.api.baseten.co/environments/production/predict \
  -H "Authorization: Api-Key YOUR_BASETEN_KEY" \
  -H "Content-Type: application/json" \
  -d @truss/test_mode2.json
```

### Cost

- Migas on T4: ~$0.63/hr (scales to zero when idle)
- LLM calls: per-token pricing via Baseten Model APIs

---

## Option 3: Docker + vLLM (self-hosted LLM)

Two containers: Migas on one GPU, vLLM on others.

```bash
MIGAS_GPU=0 VLLM_GPUS=1,2,3,4 VLLM_TENSOR_PARALLEL_SIZE=4 \
  docker compose -f docker-compose.yml -f docker-compose.vllm.yml up -d
```

**IMPORTANT:** `VLLM_TENSOR_PARALLEL_SIZE` must match the number of GPUs in `VLLM_GPUS`.

### Choosing a vLLM model

| Model | VRAM needed | Quality | Setup |
|-------|------------|---------|-------|
| `openai/gpt-oss-120b` | ~80 GB (2-4x A100) | Best | `VLLM_TENSOR_PARALLEL_SIZE=2` |
| `Qwen/Qwen3-8B` | ~16 GB (1x A10G) | Good | `VLLM_MODEL=Qwen/Qwen3-8B` |
| `Qwen/Qwen3-4B` | ~8 GB (1x T4) | Acceptable | `VLLM_MODEL=Qwen/Qwen3-4B` |
| `microsoft/phi-4` | ~8 GB (1x T4) | Good | `VLLM_MODEL=microsoft/phi-4` |

### vLLM environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_MODEL` | `openai/gpt-oss-120b` | HuggingFace model for vLLM |
| `VLLM_GPUS` | `1` | GPU index(es) for vLLM container |
| `VLLM_TENSOR_PARALLEL_SIZE` | `4` | Number of GPUs for tensor parallelism |
| `VLLM_MAX_MODEL_LEN` | `32768` | Max context length |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.60` | Fraction of GPU VRAM for vLLM |

---

## Option 4: Docker standalone

Migas only — use with pre-computed summaries, Bedrock, or any OpenAI-compatible API.

```bash
docker compose up -d
```

### CPU only (no GPU)

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

---

## Environment variables

### Migas (all deployment modes)

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace token (if model repo is private) |
| `MIGAS_DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |
| `MIGAS_MODEL` | `Synthefy/migas-1.5` | HuggingFace repo ID or local path |
| `MIGAS_GPU` | `0` | GPU index for Migas container |

### LLM (summary generation)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `bedrock` | `bedrock`, `openai`, or `anthropic` |
| `LLM_MODEL` | `anthropic.claude-3-5-haiku-20241022-v1:0` | Model ID for the chosen provider |
| `LLM_BASE_URL` | — | API base URL (required for `openai` provider, e.g. vLLM) |
| `LLM_API_KEY` | — | API key (not needed for `bedrock` — uses IAM) |
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region for Bedrock |

---

## GitHub Container Registry

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

### `GET /health` (also `GET /ping` for SageMaker)

```json
{"status": "ok", "model_loaded": true, "device": "cuda", "version": "1.5.0"}
```

### `POST /predict` (also `POST /invocations` for SageMaker)

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
{"name": "Migas-1.5", "version": "1.5.0", "llm_provider": "bedrock", "llm_model": "..."}
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
