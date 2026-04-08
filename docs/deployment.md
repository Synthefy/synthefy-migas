# Deploying Migas-1.5

Migas-1.5 ships as a Docker image with a FastAPI REST API. Two deployment modes:

| Mode | What you get | GPU needed | Summary generation |
|------|-------------|------------|-------------------|
| **Migas only** | Forecast API — you provide pre-computed summaries | 1x T4 (16 GB) | No (bring your own) |
| **Migas + vLLM** | Forecast API + automatic summary generation from text | 1x T4 + 1x A10G+ | Yes (self-contained) |

---

## Mode 1: Migas only (no vLLM)

Best for: production pipelines where summaries are pre-computed upstream, or when you already run a separate LLM server.

### Run

```bash
# GPU
docker compose up -d

# CPU (slower, no GPU required)
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

### Environment variables

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `HF_TOKEN` | — | Only for private models | HuggingFace token for model download |
| `MIGAS_API_KEY` | — | No | If set, all requests must include `X-API-Key` header |
| `MIGAS_DEVICE` | `auto` | No | `cuda`, `cpu`, or `auto` (auto-detects GPU) |
| `MIGAS_MODEL` | `Synthefy/migas-1.5` | No | HuggingFace repo ID or local path |

### Wait for the model to load

```bash
# First start downloads ~1.2 GB of model weights (cached for subsequent restarts)
curl http://localhost:8080/health
# {"status":"ok","model_loaded":true,"device":"cuda","version":"1.5.0"}
```

### Forecast with a pre-computed summary

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "dates": [
      "2024-01-02","2024-01-03","2024-01-04","2024-01-05","2024-01-08",
      "2024-01-09","2024-01-10","2024-01-11","2024-01-12","2024-01-16",
      "2024-01-17","2024-01-18","2024-01-19","2024-01-22","2024-01-23",
      "2024-01-24","2024-01-25","2024-01-26","2024-01-29","2024-01-30"
    ],
    "values": [185.3,184.1,182.7,181.9,183.4,185.0,186.2,185.8,186.5,187.1,
               186.3,188.0,189.2,190.1,191.5,192.0,191.8,192.5,193.0,193.8],
    "summaries": [
      "FACTUAL SUMMARY: Apple stock rose from 185.3 to 193.8 over 20 trading days, gaining approximately 4.6%. The rally was driven by strong iPhone 15 demand, record Services revenue, and a Q1 earnings beat on both revenue and EPS. Post-earnings, 14 analysts raised price targets, with Goldman adding Apple to its conviction buy list. China market share improved per Counterpoint data, and TSMC results confirmed healthy chip orders. Apple Vision Pro pre-orders opened to mixed reception. The buyback pace is accelerating per SEC filings.\n\nPREDICTIVE SIGNALS: Momentum is firmly bullish with accelerating buybacks, broad analyst upgrades, and WWDC 2024 as a near-term AI catalyst. Supply chain checks and TSMC results suggest continued production strength. Services revenue trajectory supports margin expansion. Upside bias likely, though Vision Pro reception and China macro remain swing factors that could moderate gains."
    ],
    "pred_len": 16
  }'
```

**Response:**

```json
{
  "forecast": [194.2, 194.8, 195.1, ...],
  "dates": ["2024-01-31", "2024-02-01", "2024-02-02", ...],
  "summaries": null,
  "univariate_forecast": null
}
```

### Ensemble averaging

Pass multiple summaries for lower-variance forecasts. The model runs one forward pass per summary and returns the median:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "dates": ["2024-01-02", "..."],
    "values": [185.3, ...],
    "summaries": [
      "FACTUAL SUMMARY: ... PREDICTIVE SIGNALS: ...",
      "FACTUAL SUMMARY: ... PREDICTIVE SIGNALS: ...",
      "FACTUAL SUMMARY: ... PREDICTIVE SIGNALS: ..."
    ],
    "pred_len": 16
  }'
```

### Get the Chronos-2 baseline too

Set `return_univariate: true` to also get the text-free univariate forecast that Migas uses internally as its baseline:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "dates": ["2024-01-02", "..."],
    "values": [185.3, ...],
    "summaries": ["FACTUAL SUMMARY: ... PREDICTIVE SIGNALS: ..."],
    "pred_len": 16,
    "return_univariate": true
  }'
```

```json
{
  "forecast": [194.2, ...],
  "dates": ["2024-01-31", ...],
  "summaries": null,
  "univariate_forecast": [194.0, ...]
}
```

---

## Mode 2: Migas + vLLM (self-contained)

Best for: end-to-end deployment where you pass raw time series with per-timestep text and the system generates summaries automatically. No external API keys needed.

### Run

```bash
docker compose -f docker-compose.yml -f docker-compose.vllm.yml up -d
```

This starts two containers:

| Container | Port | GPU | What it does |
|-----------|------|-----|-------------|
| `migas` | 8080 | 1x GPU | Loads Migas-1.5, serves `/predict` |
| `vllm` | 8004 | Remaining GPUs | Runs an LLM for summary generation |

The Migas container waits for vLLM to pass its health check before starting.

### Environment variables

All variables from Mode 1, plus:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_MODEL` | `openai/gpt-oss-120b` | LLM model for summary generation |
| `VLLM_MAX_MODEL_LEN` | `32768` | Max context length |
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for vLLM (increase for large models) |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.60` | Fraction of GPU VRAM for vLLM |

### Forecast with per-timestep text (summaries generated automatically)

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "series_name": "AAPL",
    "dates": [
      "2024-01-02","2024-01-03","2024-01-04","2024-01-05","2024-01-08",
      "2024-01-09","2024-01-10","2024-01-11","2024-01-12","2024-01-16",
      "2024-01-17","2024-01-18","2024-01-19","2024-01-22","2024-01-23",
      "2024-01-24","2024-01-25","2024-01-26","2024-01-29","2024-01-30"
    ],
    "values": [185.3,184.1,182.7,181.9,183.4,185.0,186.2,185.8,186.5,187.1,
               186.3,188.0,189.2,190.1,191.5,192.0,191.8,192.5,193.0,193.8],
    "text": [
      "Apple reports strong iPhone 15 demand in holiday quarter",
      "Tech stocks dip on broader market selloff",
      "Analysts cut near-term estimates citing China weakness",
      "Apple Vision Pro pre-orders open, mixed analyst reception",
      "Positive supply chain checks suggest iPhone production ramp",
      "Services revenue expected to hit new record",
      "Apple announces expanded AI research partnership",
      "Morgan Stanley reiterates overweight, raises PT",
      "App Store spending up 12% YoY per Sensor Tower",
      "Martin Luther King Day — markets closed",
      "Earnings whisper numbers above consensus",
      "iPad refresh rumors boost accessory supplier stocks",
      "Apple car project reportedly scaled back to focus on AI",
      "Strong results from TSMC hint at healthy Apple chip orders",
      "Q1 earnings beat on revenue and EPS, Services at record",
      "Post-earnings: 14 analysts raise price targets",
      "Goldman adds Apple to conviction buy list",
      "China iPhone market share gains per Counterpoint data",
      "Buyback pace accelerating per SEC filings",
      "Apple announces WWDC 2024 date, AI features expected"
    ],
    "pred_len": 16,
    "n_summaries": 5
  }'
```

**Response:**

```json
{
  "forecast": [194.2, 194.8, 195.1, ...],
  "dates": ["2024-01-31", "2024-02-01", "2024-02-02", ...],
  "summaries": [
    "FACTUAL SUMMARY: Apple stock rose from 185.3 to 193.8 ... PREDICTIVE SIGNALS: Momentum is bullish ...",
    "FACTUAL SUMMARY: Over the 20-day window ... PREDICTIVE SIGNALS: Near-term catalysts include ...",
    "FACTUAL SUMMARY: AAPL gained 4.6% ... PREDICTIVE SIGNALS: Analyst consensus shifted ...",
    "FACTUAL SUMMARY: The period saw ... PREDICTIVE SIGNALS: Forward-looking indicators ...",
    "FACTUAL SUMMARY: Apple shares ... PREDICTIVE SIGNALS: Key drivers ahead ..."
  ],
  "univariate_forecast": null
}
```

The generated summaries are returned in the response so you can cache and reuse them (switch to Mode 1 for subsequent calls with the same data).

### Mode 2 also supports pre-computed summaries

Even with vLLM running, you can still pass `summaries` instead of `text` to skip summary generation — the vLLM server is simply not called.

---

## Choosing a vLLM model

The default `openai/gpt-oss-120b` requires multi-GPU setups. Smaller alternatives:

| Model | VRAM needed | Quality | Setup |
|-------|------------|---------|-------|
| `openai/gpt-oss-120b` | ~80 GB (2-4x A100) | Best | `VLLM_TENSOR_PARALLEL_SIZE=2` |
| `Qwen/Qwen3-8B` | ~16 GB (1x A10G) | Good | `VLLM_MODEL=Qwen/Qwen3-8B` |
| `Qwen/Qwen3-4B` | ~8 GB (1x T4) | Acceptable | `VLLM_MODEL=Qwen/Qwen3-4B` |
| `microsoft/phi-4` | ~8 GB (1x T4) | Good | `VLLM_MODEL=microsoft/phi-4` |

Override with:

```bash
VLLM_MODEL=Qwen/Qwen3-8B docker compose -f docker-compose.yml -f docker-compose.vllm.yml up -d
```

---

## GPU sizing guide

### Mode 1 (Migas only)

Migas-1.5 inference uses ~2 GB VRAM (Chronos-2 + FinBERT + fusion head, all float32).

| Instance | VRAM | Cost | Verdict |
|----------|------|------|---------|
| T4 (16 GB) | 16 GB | ~$0.63/hr | Best fit — 8x headroom |
| L4 (24 GB) | 24 GB | ~$0.85/hr | Overkill |
| A10G (24 GB) | 24 GB | ~$1.21/hr | Overkill |

**Recommendation: T4**

### Mode 2 (Migas + vLLM)

You need GPUs for both containers. Example setups:

| Setup | Migas GPU | vLLM GPU | vLLM model | Total cost |
|-------|-----------|----------|------------|------------|
| Budget | 1x T4 | 1x T4 | Qwen3-4B | ~$1.26/hr |
| Balanced | 1x T4 | 1x A10G | Qwen3-8B | ~$1.84/hr |
| Best quality | 1x T4 | 2x A100 | gpt-oss-120b | ~$8.63/hr |

---

## Cloud deployment

### Docker (any cloud)

```bash
# Build and push
docker build -t your-registry/migas:latest .
docker push your-registry/migas:latest

# Run on remote host
docker run -d -p 8080:8080 \
  -e MIGAS_DEVICE=cuda \
  -v model-cache:/app/.cache/huggingface \
  --gpus 1 \
  your-registry/migas:latest
```

### GitHub Container Registry

Push to `main` or tag a release — the GitHub Actions workflow builds and pushes automatically:

```bash
docker pull ghcr.io/synthefy/migas-1.5:latest
docker run -d -p 8080:8080 --gpus 1 ghcr.io/synthefy/migas-1.5:latest
```

### Baseten

Use Custom Server deployment with your Docker image:

```yaml
# config.yaml
base_image:
  image: ghcr.io/synthefy/migas-1.5:latest
docker_server:
  start_command: "python3 -m uvicorn migaseval.api.main:app --host 0.0.0.0 --port 8080"
  server_port: 8080
  predict_endpoint: /predict
  readiness_endpoint: /health
  liveness_endpoint: /health
resources:
  instance_type: "T4:4x16"
```

---

## API reference

### `GET /health`

Returns model load status.

```json
{"status": "ok", "model_loaded": true, "device": "cuda", "version": "1.5.0"}
```

### `POST /predict`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `dates` | `list[str]` | Yes | — | Date strings (YYYY-MM-DD) |
| `values` | `list[float]` | Yes | — | Time series values |
| `text` | `list[str]` | Mode 1 | — | Per-timestep text. Triggers vLLM summary generation |
| `summaries` | `list[str] \| str` | Mode 2 | — | Pre-computed summary. Skips vLLM |
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

Set `MIGAS_API_KEY` to require an API key on all `/predict` requests:

```bash
MIGAS_API_KEY=my-secret-key docker compose up -d
```

```bash
curl -X POST http://localhost:8080/predict \
  -H "X-API-Key: my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```

When `MIGAS_API_KEY` is not set, authentication is disabled.

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
