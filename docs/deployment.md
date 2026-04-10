# Deploying Migas-1.5

## Overview

Migas-1.5 is a text-conditioned time-series forecasting model. It takes historical values + a text summary describing market context, and produces a forecast that responds to the narrative.

The model has two inference modes:
- **Mode 1 (text in)**: You provide raw per-timestep text (headlines, notes). An LLM generates a structured summary, then Migas forecasts using it.
- **Mode 2 (summary in)**: You provide a pre-computed summary directly. No LLM needed — Migas forecasts immediately.

## Architecture

```
┌─────────────┐     ┌───────────────┐     ┌─────────────────┐
│  Your App   │────>│  Migas API    │────>│  AWS Bedrock     │
│  (client)   │<────│  (SageMaker)  │<────│  (LLM, Mode 1)  │
└─────────────┘     └───────────────┘     └─────────────────┘
                           │
                    ┌──────┴──────┐
                    │  Migas-1.5  │
                    │  Chronos-2  │
                    │  FinBERT    │
                    └─────────────┘
```

- **Migas API** runs as a SageMaker real-time endpoint on a CPU instance (~$0.23/hr)
- **AWS Bedrock** handles LLM calls for Mode 1 (summary generation). No self-hosted LLM needed. You can use any Bedrock model (Claude, Llama, Mistral, etc.)
- **Mode 2 skips Bedrock entirely** — the summary is provided by the caller, so inference is just Migas

The model itself (~2 GB) includes Chronos-2 (time-series backbone), FinBERT (text embedder), and the fusion head. It runs fine on CPU.

## What gets deployed

Three AWS resources, managed by OpenTofu:

| Resource | What it is | Cost |
|----------|-----------|------|
| **ECR repository** | Stores the Docker image | ~$0.30/month (storage) |
| **SageMaker endpoint** | Runs the container, serves `/ping` and `/invocations` | ~$0.23/hr while running |
| **IAM role** | Lets SageMaker pull from ECR and call Bedrock | Free |

The ECR repo is created once and stays. The SageMaker endpoint can be created/destroyed freely to control costs.

## Infrastructure-as-Code layout

Everything lives in the internal-iac repo:

```
internal-iac/opentofu-iac/modules/migas-sagemaker/
├── ecr/                     # Step 1: ECR repository (deploy once)
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   ├── dev.tfvars
│   ├── prod.tfvars
│   ├── backend-dev.conf
│   └── backend-prod.conf
├── main.tf                  # Step 2: SageMaker endpoint
├── variables.tf
├── outputs.tf
├── dev.tfvars
├── prod.tfvars
├── backend-dev.conf
└── backend-prod.conf
```

**Why two steps?** The ECR repo must exist before you can push a Docker image to it, and the image must exist before SageMaker can pull it. There's a manual `docker push` step in between that OpenTofu can't do.

```
tofu apply (ECR)  →  docker push (image)  →  tofu apply (endpoint)
```

## Prerequisites

- **OpenTofu** installed (`sudo snap install opentofu --classic`)
- **AWS CLI** configured with credentials that can create SageMaker/ECR/IAM resources
- **Docker** for building and pushing the image

## Deployment steps

### 1. Create the ECR repository (one-time)

```bash
cd internal-iac/opentofu-iac/modules/migas-sagemaker/ecr
tofu init -backend-config=backend-dev.conf
tofu plan -var-file=dev.tfvars -out plan.out
tofu apply plan.out
```

This creates the ECR repo `synthefy-dev-migas-1-5` in us-east-2.

### 2. Build and push the Docker image

```bash
# Build
cd synthefy-migas
docker compose -f docker-compose.yml build

# Login to ECR
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 857077105278.dkr.ecr.us-east-2.amazonaws.com

# Tag and push
docker tag synthefy-migas-migas 857077105278.dkr.ecr.us-east-2.amazonaws.com/synthefy-dev-migas-1-5:latest
docker push 857077105278.dkr.ecr.us-east-2.amazonaws.com/synthefy-dev-migas-1-5:latest
```

### 3. Deploy the SageMaker endpoint

```bash
cd internal-iac/opentofu-iac/modules/migas-sagemaker
tofu init -backend-config=backend-dev.conf
tofu plan -var-file=dev.tfvars -out plan.out
tofu apply plan.out
```

Takes ~5-10 minutes. SageMaker provisions a VM, pulls the image, downloads model weights from HuggingFace, and starts the server.

Check status:
```bash
aws sagemaker describe-endpoint --endpoint-name migas-sagemaker-dev --region us-east-2 --query "EndpointStatus" --output text
```

Wait for `InService`.

### 4. Test

Test JSON bodies are in the repo under `test_*.json`. Use `file://` to avoid shell escaping issues with long JSON.

**Mode 2 — pre-computed summary (no LLM call):**
```bash
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name migas-sagemaker-dev \
  --region us-east-2 \
  --content-type application/json \
  --body file://test_mode2.json \
  /dev/stdout
```

**Mode 1 — text in, Bedrock generates summary (default Haiku 4.5):**
```bash
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name migas-sagemaker-dev \
  --region us-east-2 \
  --content-type application/json \
  --body file://test_mode1.json \
  /dev/stdout
```

**Mode 1 — with per-request LLM override (Sonnet):**
```bash
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name migas-sagemaker-dev \
  --region us-east-2 \
  --content-type application/json \
  --body file://test_mode1_sonnet.json \
  /dev/stdout
```

Mode 2 returns instantly. Mode 1 takes a few seconds (LLM call to Bedrock).

**Tip:** For inline testing with short payloads, avoid multiline strings in the terminal — shell line wrapping injects literal newlines into JSON strings, causing parse errors. Always use `file://` for payloads with special characters.

## Day-to-day operations

### Pushing a code update

When you change the Migas code and want to deploy it:

```bash
# Rebuild and push (use a new tag to force SageMaker to pull fresh)
cd synthefy-migas
docker compose -f docker-compose.yml build
docker tag synthefy-migas-migas 857077105278.dkr.ecr.us-east-2.amazonaws.com/synthefy-dev-migas-1-5:v4
docker push 857077105278.dkr.ecr.us-east-2.amazonaws.com/synthefy-dev-migas-1-5:v4

# Redeploy endpoint with new tag
cd internal-iac/opentofu-iac/modules/migas-sagemaker
tofu destroy -var-file=dev.tfvars
tofu plan -var-file=dev.tfvars -var="image_tag=v4" -out plan.out
tofu apply plan.out
```

**Important:** Use a unique image tag (v2, v3, v4...) for each deploy. SageMaker caches `latest` and won't pull a new image unless the tag changes. Destroy + recreate is needed to pick up a new model definition.

### Stop the endpoint (stops billing)

```bash
cd internal-iac/opentofu-iac/modules/migas-sagemaker
tofu destroy -var-file=dev.tfvars
```

The ECR repo and image stay intact. Only the SageMaker endpoint is removed.

### Restart the endpoint

```bash
cd internal-iac/opentofu-iac/modules/migas-sagemaker
tofu plan -var-file=dev.tfvars -out plan.out
tofu apply plan.out
```

### Check logs

```bash
# Find log streams
aws logs describe-log-streams \
  --log-group-name /aws/sagemaker/Endpoints/migas-sagemaker-dev \
  --region us-east-2 --order-by LastEventTime --descending --limit 1

# Read logs
aws logs get-log-events \
  --log-group-name /aws/sagemaker/Endpoints/migas-sagemaker-dev \
  --log-stream-name <STREAM_NAME> \
  --region us-east-2 --limit 50
```

## Configuration

### Environment variables (set in dev.tfvars / prod.tfvars)

| Variable | Default | Description |
|----------|---------|-------------|
| `instance_type` | `ml.m5.xlarge` | SageMaker instance type (CPU) |
| `instance_count` | `1` | Number of instances |
| `llm_provider` | `bedrock` | LLM provider for Mode 1 |
| `llm_model` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Bedrock model ID (use `us.` prefix) |
| `migas_model` | `Synthefy/migas-1.5` | HuggingFace model ID |
| `image_tag` | `latest` | Docker image tag to deploy |

### Changing the Bedrock model

Edit `dev.tfvars`:
```hcl
llm_model = "us.meta.llama3-3-70b-instruct-v1:0"
```

Then `tofu plan` + `tofu apply`. Or override per-request by passing `llm_model` in the JSON body.

Bedrock model IDs require a `us.` prefix for cross-region inference profiles. Available models:

| Model | Bedrock ID |
|-------|-----------|
| Claude Haiku 4.5 (default) | `us.anthropic.claude-haiku-4-5-20251001-v1:0` |
| Claude Sonnet 4.6 | `us.anthropic.claude-sonnet-4-6` |
| Claude Sonnet 4 | `us.anthropic.claude-sonnet-4-20250514-v1:0` |
| DeepSeek V3.2 | `us.deepseek.v3.2` |
| Llama 3.3 70B | `us.meta.llama3-3-70b-instruct-v1:0` |
| Mistral Large 3 | `us.mistral.mistral-large-3-675b-instruct` |

List all active models: `aws bedrock list-foundation-models --region us-east-2 --query "modelSummaries[?modelLifecycle.status=='ACTIVE'].modelId"`

### Production deployment

Same process, use `prod.tfvars` and `backend-prod.conf` everywhere instead of `dev`.

## API reference

The SageMaker endpoint exposes two routes (SageMaker convention):

### `GET /ping`

Health check. Returns HTTP 200 when the model is loaded.

```json
{"status": "ok", "model_loaded": true, "device": "cpu", "version": "1.5.0"}
```

### `POST /invocations`

Inference endpoint. Same schema as `/predict`.

**Request:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `dates` | `list[str]` | Yes | - | Date strings (YYYY-MM-DD) |
| `values` | `list[float]` | Yes | - | Time series values |
| `text` | `list[str]` | Mode 1 | - | Per-timestep text. Triggers Bedrock summary generation |
| `summaries` | `list[str] \| str` | Mode 2 | - | Pre-computed summary. Skips LLM |
| `series_name` | `str` | No | `"series"` | Name used in summary generation prompt |
| `pred_len` | `int` | No | `16` | Forecast horizon (1-16) |
| `n_summaries` | `int` | No | `5` | Ensemble size (Mode 1 only) |
| `llm_model` | `str` | No | server default | Override Bedrock model per-request (e.g. `us.anthropic.claude-sonnet-4-6`) |
| `return_univariate` | `bool` | No | `false` | Also return Chronos-2 baseline |

Either `text` or `summaries` must be provided.

**Response:**

```json
{
  "forecast": [2.44, 2.45, ...],
  "dates": ["2020-03-17", "2020-03-18", ...],
  "summaries": null,
  "univariate_forecast": null
}
```

## Summary format

Summaries must contain exactly two sections:

```
FACTUAL SUMMARY:
[Dense analytical paragraph about observed facts, price moves, catalysts]

PREDICTIVE SIGNALS:
[Forward-looking paragraph — relative terms only, no absolute price targets]
```

This is the format Migas-1.5 was trained on. Deviating from it degrades forecast quality.

## Docker stacks (local development)

For local testing without SageMaker, the repo includes Docker Compose stacks:

| Stack | Command | Description |
|-------|---------|-------------|
| Standalone | `docker compose up -d` | Migas only, defaults to Bedrock |
| + vLLM | `docker compose -f docker-compose.yml -f docker-compose.vllm.yml up -d` | Adds self-hosted vLLM sidecar |
| CPU only | `docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d` | No GPU needed |

The standalone stack listens on `localhost:8080` with the same `/predict` API.

## Cost summary

| Component | Cost | When |
|-----------|------|------|
| ECR storage | ~$0.30/month | Always (image stored) |
| SageMaker ml.m5.xlarge | ~$0.23/hr | Only while endpoint is running |
| Bedrock (Claude Haiku 4.5) | ~$0.80/1M input tokens | Only Mode 1 calls |

Destroy the SageMaker endpoint when not in use to avoid charges. ECR cost is negligible.
