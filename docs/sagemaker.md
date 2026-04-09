# Deploying Migas-1.5 on AWS SageMaker

This document explains how to deploy Migas-1.5 as a real-time inference endpoint
on AWS SageMaker. It assumes no prior AWS experience and covers everything from
core concepts to production deployment.

---

## Table of contents

1. [AWS concepts you need to know](#1-aws-concepts-you-need-to-know)
2. [How SageMaker inference works](#2-how-sagemaker-inference-works)
3. [Where Migas stands today](#3-where-migas-stands-today)
4. [Deployment paths: Custom container vs Marketplace](#4-deployment-paths-custom-container-vs-marketplace)
5. [Implementation plan: Custom container](#5-implementation-plan-custom-container)
6. [Code changes required](#6-code-changes-required)
7. [Deployment workflow](#7-deployment-workflow)
8. [Instance types and cost](#8-instance-types-and-cost)
9. [Summary generation on SageMaker](#9-summary-generation-on-sagemaker)
10. [SageMaker Marketplace (Phase 2)](#10-sagemaker-marketplace-phase-2)
11. [Comparison with existing deployments](#11-comparison-with-existing-deployments)
12. [FAQ](#12-faq)

---

## 1. AWS concepts you need to know

Before diving into SageMaker, here are the AWS services involved and what they do:

### AWS account and IAM

- **AWS Account** — Your billing and resource container. Everything you create lives
  inside an account. You sign up at aws.amazon.com with an email and credit card.
- **IAM (Identity and Access Management)** — Controls *who* can do *what* in your
  account. You create "roles" with specific permissions. SageMaker needs a role that
  lets it pull containers, read S3, and manage endpoints.
- **IAM Execution Role** — A role that SageMaker *assumes* when running your model.
  It needs permissions for ECR (pull images), S3 (read model artifacts), and
  CloudWatch (write logs). AWS provides a managed policy called
  `AmazonSageMakerFullAccess` that covers most of this.

### ECR (Elastic Container Registry)

- AWS's Docker Hub equivalent. You push your Docker image here, and SageMaker pulls
  it when creating an endpoint.
- Images are stored in "repositories" (e.g., `migas-1.5`).
- Each image has a URI like: `123456789012.dkr.ecr.us-east-1.amazonaws.com/migas-1.5:latest`

### S3 (Simple Storage Service)

- AWS's file storage. Used for storing model artifacts (weights) if you don't bake
  them into the container.
- Files are organized in "buckets" (e.g., `s3://my-migas-bucket/model.tar.gz`).
- SageMaker can automatically download model artifacts from S3 into the container
  at `/opt/ml/model/`.

### SageMaker

- AWS's ML platform. We only use its **real-time inference** feature:
  - **Model** — A record that points to your container image (ECR) and optionally
    model data (S3). It's metadata, not a running thing.
  - **Endpoint Configuration** — Specifies which instance type, how many instances,
    and which model to use. Think of it as a "deployment plan."
  - **Endpoint** — The actual running service. Has a URL you can invoke.
    Creating an endpoint provisions GPU instances and starts your container.

### CloudWatch

- AWS's logging and monitoring service. SageMaker sends container stdout/stderr here
  automatically. You'll use it to debug startup issues.

### The flow

```
You build Docker image
    → push to ECR
        → create SageMaker Model (pointing to ECR image)
            → create Endpoint Config (instance type, count)
                → create Endpoint (provisions GPU, starts container)
                    → invoke via boto3 or HTTPS
```

---

## 2. How SageMaker inference works

SageMaker runs your Docker container on a GPU instance and routes traffic to it.
Your container must follow a simple contract:

### Container contract

| Requirement | Details |
|-------------|---------|
| **Port** | Listen on **8080** (hardcoded, not configurable) |
| **`GET /ping`** | Health check. Return HTTP 200 when ready. Called every few seconds. Must respond within 2 seconds. |
| **`POST /invocations`** | Inference endpoint. Receives the request body, returns predictions. Must respond within 60 seconds. |
| **Startup** | Container must pass health checks within **8 minutes** of starting, or endpoint creation fails. |
| **User** | Must run as root. |
| **Entry command** | SageMaker runs `docker run <image> serve`. Your container can handle the `serve` argument or ignore it. |

### Model artifacts (optional)

If you specify a `ModelDataUrl` when creating the SageMaker Model, SageMaker:
1. Downloads a `.tar.gz` file from S3
2. Extracts it into `/opt/ml/model/` in your container
3. This happens *before* your server starts

For Migas, we have two choices:
- **Option A**: Download weights from HuggingFace Hub at startup (current behavior) —
  simpler, but requires outbound network access
- **Option B**: Package weights in S3 and load from `/opt/ml/model/` —
  more "SageMaker-native," required for Marketplace

### Payload limits

| Endpoint type | Max payload | Timeout |
|---------------|-------------|---------|
| Real-time | 6 MB | 60 seconds |
| Async | 1 GB (via S3) | 15 minutes |

6 MB is plenty for time series + summaries (a typical request is ~10-50 KB).

### Invocation

Clients call the endpoint via the AWS SDK (boto3):

```python
import boto3, json

client = boto3.client("sagemaker-runtime")
response = client.invoke_endpoint(
    EndpointName="migas-1-5",
    ContentType="application/json",
    Body=json.dumps({"dates": [...], "values": [...], "summaries": "..."}),
)
result = json.loads(response["Body"].read())
```

SageMaker handles authentication (AWS IAM credentials), TLS, and load balancing.
There is no public URL — only AWS-authenticated callers can invoke the endpoint.

---

## 3. Where Migas stands today

The existing infrastructure is remarkably close to SageMaker-ready:

| SageMaker requirement | Migas current state | Gap |
|---|---|---|
| Container on port 8080 | Dockerfile already uses port 8080 | **None** |
| `GET /ping` returning 200 | Has `GET /health` returning JSON | Add alias |
| `POST /invocations` | Has `POST /predict` | Add alias |
| NVIDIA GPU container | CUDA 12.4.1 runtime image | **None** |
| JSON request/response | FastAPI + Pydantic | **None** |
| Stateless inference | Model loaded once at startup | **None** |
| Responds within 60s | First request ~30-60s (model warm-up), then <2s | May need warm-up strategy |
| Health check within 8 min | Model loads in ~60-120s on T4 | **Fine** |

**The core code changes are ~20 lines of Python.** The rest is deployment scripting.

---

## 4. Deployment paths: Custom container vs Marketplace

There are two ways to get Migas on SageMaker:

### Path 1: Custom container (your own inference code)

AWS calls this "using your own inference code" — you build a Docker image with your
model and server, push it to ECR, and deploy it as a SageMaker endpoint. You'll
sometimes hear this called "BYOC" (bring your own container) in blog posts and
community discussions, but the official AWS term is just "custom inference container."

**Who it's for**: Your team, your customers on your AWS account, or customers who
pull your image into their own accounts.

**Pros**:
- Fast to implement (days, not weeks)
- Full control over the container, updates, and configuration
- No approval process
- Can use outbound network (HuggingFace download, vLLM sidecar, etc.)

**Cons**:
- No marketplace discovery — customers must set it up manually
- You manage billing relationships yourself

**This is the recommended starting point.**

### Path 2: AWS Marketplace listing

**What it is**: You list Migas-1.5 as a product on the AWS Marketplace. Customers
subscribe, and SageMaker deploys it for them with one click.

**Who it's for**: Commercial distribution to external customers (like TabPFN did).

**Pros**:
- Discovery via AWS Marketplace (searchable by any AWS customer)
- AWS handles billing, metering, and payments
- One-click deployment for customers
- Professional credibility

**Cons**:
- Requires AWS Marketplace Seller Registration (tax info, bank account for payouts)
- Review process takes 2-4 weeks
- **No outbound network** — container cannot call HuggingFace, vLLM, or any external API
- Must bundle all model weights inside the container or S3 artifact
- Must set pricing (per-hour, per-inference, or free tier)

**Build on top of Path 1 once it's validated.**

---

## 5. Implementation plan: Custom container

### Phase 1: Make the container SageMaker-compatible

**Task 1 — Add SageMaker route aliases (~20 lines)**

Add `/ping` and `/invocations` endpoints to the existing FastAPI app. These are
simple aliases that reuse existing logic. The current `/health` and `/predict`
endpoints remain unchanged for non-SageMaker use.

Files to modify:
- `src/migaseval/api/main.py`

**Task 2 — Handle the `serve` argument**

SageMaker runs `docker run <image> serve`. The current `CMD` doesn't handle this.
Two options:
- Add a small entrypoint script that ignores the `serve` argument and starts uvicorn
- Or set `ENTRYPOINT` + `CMD` so the container works with or without `serve`

Files to create/modify:
- `sagemaker/serve` (entrypoint script)
- `Dockerfile` (add SageMaker-compatible entrypoint as an option, or handle in a
  SageMaker-specific Dockerfile)

**Task 3 — Support model loading from `/opt/ml/model/`**

If model weights exist at `/opt/ml/model/model.pt`, load from there instead of
downloading from HuggingFace. This is optional for custom containers but required for Marketplace.

Files to modify:
- `src/migaseval/api/main.py` (check `/opt/ml/model/` before HF download)

### Phase 2: Deployment infrastructure

**Task 4 — ECR push script**

A script that builds the Docker image and pushes it to ECR.

Files to create:
- `sagemaker/push_to_ecr.sh`

**Task 5 — SageMaker deployment script**

A Python script using boto3 that:
1. Creates a SageMaker Model pointing to the ECR image
2. Creates an Endpoint Configuration with instance type
3. Creates an Endpoint
4. Waits for it to become `InService`

Files to create:
- `sagemaker/deploy.py`

**Task 6 — Example invocation script**

A Python script showing how to call the deployed endpoint.

Files to create:
- `sagemaker/invoke_example.py`

### Phase 3: Testing and documentation

**Task 7 — Local testing with SageMaker Local Mode**

SageMaker Local Mode lets you test the container locally before deploying to AWS.
This catches contract violations (wrong port, missing endpoints) without waiting
for real provisioning.

**Task 8 — CI/CD (optional)**

GitHub Actions workflow to build and push to ECR on release, similar to the
existing GHCR workflow.

### Summary of new/modified files

```
sagemaker/
  deploy.py              # boto3 deployment script
  invoke_example.py      # example client code
  push_to_ecr.sh         # build + push Docker image to ECR
  serve                   # entrypoint script for SageMaker
  Dockerfile.sagemaker   # (optional) SageMaker-specific Dockerfile
  README.md              # quick start for SageMaker deployment

src/migaseval/api/
  main.py                # MODIFIED: add /ping and /invocations aliases
```

---

## 6. Code changes required

### 6.1 FastAPI route aliases (`src/migaseval/api/main.py`)

```python
# SageMaker health check — must return 200 when model is ready
@app.get("/ping")
def ping():
    if _pipeline is None:
        raise HTTPException(503, "Model not loaded")
    return Response(status_code=200)

# SageMaker inference endpoint — same as /predict
@app.post("/invocations", response_model=PredictResponse)
def invocations(req: PredictRequest):
    return predict(req)
```

### 6.2 Model loading with `/opt/ml/model/` fallback

```python
# In lifespan(), before from_pretrained:
model_source = os.environ.get("MIGAS_MODEL", "Synthefy/migas-1.5")
local_model = "/opt/ml/model/model.pt"
if os.path.exists(local_model):
    model_source = local_model  # Use SageMaker-provided artifact

_pipeline = MigasPipeline.from_pretrained(model_source, ...)
```

### 6.3 Entrypoint script (`sagemaker/serve`)

```bash
#!/bin/bash
# SageMaker runs: docker run <image> serve
# This script ignores the argument and starts the server
exec python -m uvicorn migaseval.api.main:app --host 0.0.0.0 --port 8080
```

### 6.4 Deployment script (`sagemaker/deploy.py`)

```python
import boto3
import sagemaker

role = "arn:aws:iam::ACCOUNT_ID:role/SageMakerExecutionRole"
image_uri = "ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/migas-1.5:latest"

sm = boto3.client("sagemaker")

# 1. Create Model
sm.create_model(
    ModelName="migas-1-5",
    PrimaryContainer={
        "Image": image_uri,
        "Environment": {
            "MIGAS_DEVICE": "auto",
            "MIGAS_MODEL": "Synthefy/migas-1.5",
        },
    },
    ExecutionRoleArn=role,
)

# 2. Create Endpoint Config
sm.create_endpoint_config(
    EndpointConfigName="migas-1-5-config",
    ProductionVariants=[{
        "VariantName": "primary",
        "ModelName": "migas-1-5",
        "InstanceType": "ml.g4dn.xlarge",
        "InitialInstanceCount": 1,
    }],
)

# 3. Create Endpoint
sm.create_endpoint(
    EndpointName="migas-1-5",
    EndpointConfigName="migas-1-5-config",
)
# Takes 5-10 minutes to provision GPU and start container
```

### 6.5 Invocation example (`sagemaker/invoke_example.py`)

```python
import boto3, json

runtime = boto3.client("sagemaker-runtime")

payload = {
    "series_name": "crude_oil",
    "dates": ["2024-08-27", "2024-08-28", ...],
    "values": [71.87, 71.61, ...],
    "summaries": "FACTUAL SUMMARY: ... PREDICTIVE SIGNALS: ...",
    "pred_len": 16,
}

response = runtime.invoke_endpoint(
    EndpointName="migas-1-5",
    ContentType="application/json",
    Body=json.dumps(payload),
)

result = json.loads(response["Body"].read())
print(result["forecast"])  # [61.94, 62.08, ...]
print(result["dates"])     # ["2025-02-25", "2025-02-26", ...]
```

---

## 7. Deployment workflow

Step-by-step instructions for someone who hasn't used AWS before.

### 7.1 Prerequisites

```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip awscliv2.zip && sudo ./aws/install

# Configure credentials (you'll need an Access Key from IAM)
aws configure
# Enter: AWS Access Key ID, Secret Access Key, Region (e.g., us-east-1)

# Install boto3 and sagemaker SDK
pip install boto3 sagemaker
```

### 7.2 Create IAM execution role

In the AWS Console:
1. Go to **IAM** > **Roles** > **Create role**
2. Select **SageMaker** as the trusted service
3. Attach the policy `AmazonSageMakerFullAccess`
4. Name it `SageMakerExecutionRole`
5. Copy the role ARN (e.g., `arn:aws:iam::123456789012:role/SageMakerExecutionRole`)

### 7.3 Create ECR repository

```bash
aws ecr create-repository --repository-name migas-1.5

# Output includes repositoryUri — save it:
# 123456789012.dkr.ecr.us-east-1.amazonaws.com/migas-1.5
```

### 7.4 Build and push Docker image

```bash
# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    123456789012.dkr.ecr.us-east-1.amazonaws.com

# Build the image
docker build -t migas-1.5 .

# Tag for ECR
docker tag migas-1.5:latest \
    123456789012.dkr.ecr.us-east-1.amazonaws.com/migas-1.5:latest

# Push
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/migas-1.5:latest
```

### 7.5 Deploy endpoint

```bash
python sagemaker/deploy.py
# Wait 5-10 minutes for endpoint to become InService
```

### 7.6 Test

```bash
python sagemaker/invoke_example.py
```

### 7.7 Monitor

```bash
# Check endpoint status
aws sagemaker describe-endpoint --endpoint-name migas-1-5

# View logs (container stdout/stderr)
aws logs tail /aws/sagemaker/Endpoints/migas-1-5 --follow
```

### 7.8 Clean up (to stop billing)

```bash
aws sagemaker delete-endpoint --endpoint-name migas-1-5
aws sagemaker delete-endpoint-config --endpoint-config-name migas-1-5-config
aws sagemaker delete-model --model-name migas-1-5
```

**Important**: SageMaker charges per-hour while the endpoint is running, even with
no traffic. Always delete endpoints when not in use, or use async/serverless
inference for cost savings.

---

## 8. Instance types and cost

### Recommended instances for Migas-1.5

**Migas only (Mode 2: pre-computed summaries, or Bedrock for LLM)**:

| Instance | GPU | VRAM | vCPU | RAM | On-demand $/hr | Notes |
|----------|-----|------|------|-----|----------------|-------|
| `ml.g4dn.xlarge` | 1x T4 | 16 GB | 4 | 16 GB | ~$0.74 | **Best value** — same GPU as Baseten |
| `ml.g5.xlarge` | 1x A10G | 24 GB | 4 | 16 GB | ~$1.41 | Faster inference, more headroom |
| `ml.g5.2xlarge` | 1x A10G | 24 GB | 8 | 32 GB | ~$1.52 | More CPU/RAM for data processing |

**Migas + vLLM in single container (Option A)**:

| Instance | GPU | VRAM | vCPU | RAM | On-demand $/hr | vLLM model |
|----------|-----|------|------|-----|----------------|------------|
| `ml.g4dn.xlarge` | 1x T4 | 16 GB | 4 | 16 GB | ~$0.74 | Qwen3-4B (tight, ~10 GB total) |
| `ml.g5.2xlarge` | 1x A10G | 24 GB | 8 | 32 GB | ~$1.52 | **Qwen3-8B** (recommended) |
| `ml.g5.12xlarge` | 4x A10G | 96 GB | 48 | 192 GB | ~$7.09 | Qwen3-32B or larger |

**Recommendation**: For Migas + Bedrock, use `ml.g4dn.xlarge` (~$0.74/hr). For
Migas + vLLM in one container, use `ml.g5.2xlarge` (~$1.52/hr) with Qwen3-8B.

### Cost comparison with existing options

| Platform | Instance | $/hr | Auto-scaling | Scale to zero |
|----------|----------|------|--------------|---------------|
| **Baseten** | T4 | ~$0.63 | Yes | Yes |
| **SageMaker** | ml.g4dn.xlarge | ~$0.74 | Yes (manual config) | No (real-time) |
| **SageMaker Serverless** | — | ~$0.20/min active | Auto | Yes |
| **SageMaker Async** | ml.g4dn.xlarge | ~$0.74 | Yes | Yes (min=0) |
| **Docker (self-hosted)** | Depends | Depends | Manual | Manual |

### Serverless inference (alternative)

SageMaker Serverless Inference automatically scales to zero and charges only when
processing requests. It's great for low-traffic or bursty workloads but has:
- Cold start latency (30-60s when scaling from zero)
- Max 6 MB payload, 60s timeout
- Limited GPU support (currently in preview)

For Migas, real-time endpoints are more practical since GPU is required.

### Async inference (alternative for batch)

For batch workloads or long-running predictions, SageMaker Async Inference:
- Accepts requests via S3 (up to 1 GB payload)
- Scales to zero when idle (no charges)
- Results written to S3
- Up to 15-minute timeout

This could be useful for batch inference with many series.

---

## 9. Summary generation on SageMaker

Summary generation (Mode 1: per-timestep text → LLM-generated summaries) requires
an LLM server. On Docker, this is simple — `docker-compose.vllm.yml` runs vLLM as
a sidecar on `localhost:8004`. On SageMaker, it's harder because **containers in a
multi-container endpoint cannot talk to each other over localhost**. SageMaker
inference pipelines are strictly sequential (output of A → input of B), not
bidirectional.

Here are the real architecture options:

### Option A: Single container running both Migas + vLLM

Pack both Migas (FastAPI on port 8080) and vLLM (on port 8004) into one container.
Use a process manager (like `supervisord`) to start both processes. vLLM runs on
`localhost:8004` exactly like the docker-compose setup — the existing code works
unchanged.

```
┌─────────────────────────────────────────┐
│         Single SageMaker Endpoint       │
│  ┌──────────────┐  ┌─────────────────┐  │
│  │ Migas FastAPI │→ │ vLLM (Qwen3-8B) │  │
│  │   :8080       │  │   :8004         │  │
│  └──────────────┘  └─────────────────┘  │
│         ml.g5.2xlarge (1x A10G, 24 GB)  │
└─────────────────────────────────────────┘
```

**Pros**:
- Self-contained, single endpoint, no networking config
- Existing code works with zero changes (vLLM on localhost:8004)
- Simple to deploy and manage

**Cons**:
- Needs a larger GPU instance to fit both models in VRAM
  - Migas ~2 GB + Qwen3-8B ~16 GB = ~18 GB → needs A10G (24 GB) or better
  - Migas ~2 GB + Qwen3-4B ~8 GB = ~10 GB → fits on T4 (16 GB) but tight
- vLLM startup adds to the 8-minute health check window (vLLM can take 2-4 min to load)
- Cannot scale Migas and vLLM independently

**Estimated cost**: `ml.g5.2xlarge` at ~$1.52/hr (1x A10G, 24 GB VRAM, 8 vCPU, 32 GB RAM)

**VRAM budget on A10G (24 GB)**:

| Component | VRAM | Notes |
|-----------|------|-------|
| Migas (Chronos-2 + FinBERT + fusion) | ~2 GB | float32 |
| Qwen3-8B (vLLM, fp16) | ~16 GB | |
| vLLM KV cache overhead | ~3 GB | |
| **Total** | **~21 GB** | Fits in 24 GB with headroom |

### Option B: vLLM on a separate EC2 instance

Run vLLM on a standard EC2 GPU instance with a private IP. Set `VLLM_BASE_URL` to
point at it. The Migas SageMaker endpoint calls vLLM over the VPC network.

```
┌───────────────────────┐        ┌─────────────────────┐
│  SageMaker Endpoint   │  HTTP  │   EC2 Instance      │
│  Migas FastAPI :8080  │───────→│   vLLM :8004        │
│  ml.g4dn.xlarge (T4)  │        │   g4dn.xlarge (T4)  │
└───────────────────────┘        └─────────────────────┘
         VPC: same subnet or peered
```

**Pros**:
- Each component runs on its own GPU — right-sized independently
- vLLM can serve multiple clients (not just Migas)
- Cheaper GPU options (EC2 spot instances, reserved instances)

**Cons**:
- Two things to manage (SageMaker endpoint + EC2 instance)
- Requires VPC/subnet/security group configuration:
  - Both must be in the same VPC (or use VPC peering)
  - Security group must allow inbound TCP 8004 from the SageMaker subnet
  - SageMaker endpoint needs VPC config (`VpcConfig` in `CreateModel`)
- Network latency between instances (typically <1ms within same AZ)
- EC2 doesn't auto-scale or scale-to-zero (you manage uptime)

**VPC setup** (one-time):
1. Create or use an existing VPC with a private subnet
2. Create a security group allowing port 8004 from the SageMaker subnet
3. Launch EC2 with GPU in that subnet
4. When creating the SageMaker Model, pass `VpcConfig` with the same subnets
   and security groups so the container can reach EC2

**Estimated cost**: ~$0.74/hr (Migas on g4dn.xlarge) + ~$0.74/hr (vLLM on EC2 g4dn.xlarge) = ~$1.48/hr

### Option C: AWS Bedrock instead of vLLM (most AWS-native)

Replace vLLM with **Amazon Bedrock** — AWS's managed LLM service. Bedrock offers
Claude, Llama, Qwen3, Mistral, and others as serverless APIs. No GPU to manage,
no server to run. The Migas container calls Bedrock via `boto3`.

```
┌───────────────────────┐        ┌─────────────────────┐
│  SageMaker Endpoint   │ boto3  │   Amazon Bedrock     │
│  Migas FastAPI :8080  │───────→│   (Qwen3-32B, etc.) │
│  ml.g4dn.xlarge (T4)  │        │   Serverless         │
└───────────────────────┘        └─────────────────────┘
        IAM role needs bedrock:InvokeModel permission
```

**Pros**:
- No LLM server to deploy, manage, or pay for when idle
- Bedrock is serverless — pay per token, auto-scales, zero infrastructure
- Available models include Qwen3 (32B, 235B), Claude, Llama 4, Mistral
- No VPC config needed — Bedrock is accessed via AWS SDK, not HTTP
- Single SageMaker endpoint, small GPU instance
- Most "AWS-native" solution

**Cons**:
- Requires adding a `"bedrock"` provider to `summary_utils.py` (~30 lines of code)
- Per-token pricing can be expensive at high volume (vs fixed GPU cost for vLLM)
- Slightly higher latency than localhost vLLM (~1-3s per call)
- Model selection limited to what Bedrock offers (but Qwen3, Claude, Llama are there)
- No web search capability (unlike Anthropic's direct API with web_search tool)

**Code change required** — add Bedrock as a provider in `call_llm()`:

```python
elif provider == "bedrock":
    import boto3
    client = boto3.client("bedrock-runtime")
    resp = client.converse(
        modelId=model or "us.amazon.nova-lite-v1:0",
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens or 2048, "temperature": 0.3},
    )
    return resp["output"]["message"]["content"][0]["text"].strip()
```

**Estimated cost**: ~$0.74/hr (Migas on g4dn.xlarge) + ~$0.001-0.01 per summary
(Bedrock token pricing varies by model). Much cheaper at low volume.

**Bedrock model pricing examples** (per 1M tokens):

| Model | Input | Output | Notes |
|-------|-------|--------|-------|
| Amazon Nova Lite | $0.06 | $0.24 | Cheapest, good quality |
| Amazon Nova Pro | $0.80 | $3.20 | Better quality |
| Qwen3-32B | $0.20 | $0.20 | Good for summarization |
| Claude Haiku 4.5 | $0.80 | $4.00 | High quality |
| Llama 4 Scout | $0.17 | $0.17 | Good balance |

A typical summary generation call uses ~2K input + ~500 output tokens.
At 5 summaries per request, that's ~12.5K tokens total — roughly $0.001-0.05
per request depending on model.

### Option D: Pre-computed summaries only

Users always provide `summaries` in the request. No LLM needed.

**Pros**: Simplest deployment, single small GPU, no extra infrastructure
**Cons**: Users must generate summaries themselves before calling the endpoint

### Recommendation

**Option C (Bedrock)** is the best fit for SageMaker:
- Single endpoint, small GPU, no LLM infra to manage
- ~30 lines of code to add Bedrock as a provider
- Pay-per-use pricing (no idle GPU costs for the LLM)
- Most natural for AWS customers

**Option A (single container)** is the best fit if you want self-contained
inference with no external dependencies — same architecture as docker-compose,
just in one container.

Both options support Mode 1 (text → summaries → forecast) end-to-end.

---

## 10. SageMaker Marketplace (Phase 2)

This section covers what's needed if you want to list Migas like TabPFN did.

### What TabPFN did

TabPFN is listed on the AWS Marketplace as a subscribable model. Customers:
1. Subscribe to the listing (agree to terms, pricing)
2. Deploy an endpoint from the SageMaker console
3. Invoke it via `boto3`

They charge per-hour for the compute instance.

### Steps to list on Marketplace

1. **Register as an AWS Marketplace Seller**
   - Go to [AWS Marketplace Management Portal](https://aws.amazon.com/marketplace/management/)
   - Provide business details, tax information, bank account for payouts
   - Takes a few business days to get approved

2. **Prepare the container for Marketplace**

   Marketplace containers have **additional restrictions**:
   - **No outbound network access** — cannot call HuggingFace, APIs, or the internet
   - Model weights must be bundled in the container or provided via S3 model package
   - Container must work completely offline

   This means:
   - Bake model weights into the Docker image (or use S3 model artifact)
   - Pre-download Chronos-2 and FinBERT weights into the image at build time
   - Only Mode 2 (pre-computed summaries) is supported
   - No HuggingFace Hub downloads at runtime

3. **Create a Model Package**

   A "Model Package" is the Marketplace-specific wrapper:
   ```python
   sm.create_model_package(
       ModelPackageGroupName="migas-1-5",
       InferenceSpecification={
           "Containers": [{
               "Image": ecr_image_uri,
               "ModelDataUrl": "s3://bucket/model.tar.gz",  # optional
           }],
           "SupportedInstanceTypes": ["ml.g4dn.xlarge", "ml.g5.xlarge"],
           "SupportedContentTypes": ["application/json"],
           "SupportedResponseMIMETypes": ["application/json"],
       },
   )
   ```

4. **Create the Marketplace listing**
   - Set pricing: free, hourly, or per-inference
   - Write description, usage instructions, sample code
   - Specify supported regions and instance types
   - Submit for review (2-4 weeks)

5. **Validation**

   AWS runs automated tests:
   - Container must start and pass `/ping` within 8 minutes
   - Must handle `/invocations` correctly
   - Must not make outbound network calls
   - Must handle error cases gracefully

### Marketplace-specific Dockerfile changes

```dockerfile
# Pre-download all model weights at build time
RUN python -c "
from huggingface_hub import hf_hub_download
# Migas fusion weights
hf_hub_download('Synthefy/migas-1.5', 'model.pt', cache_dir='/app/models')
# Chronos-2 weights (will be cached)
from chronos import ChronosPipeline
ChronosPipeline.from_pretrained('amazon/chronos-2', cache_dir='/app/models')
# FinBERT weights
from sentence_transformers import SentenceTransformer
SentenceTransformer('ProsusAI/finbert', cache_dir='/app/models')
"

ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
```

This makes the Docker image larger (~2-3 GB) but fully self-contained.

---

## 11. Comparison with existing deployments

| Aspect | Docker (self-hosted) | Baseten (Truss) | SageMaker (custom) | SageMaker Marketplace |
|--------|---------------------|-----------------|-------------------|-----------------------|
| **Setup effort** | Low (docker compose) | Low (truss push) | Medium (ECR + boto3) | High (seller registration) |
| **Infrastructure** | You manage | Baseten manages | AWS manages compute | AWS manages everything |
| **GPU provisioning** | Manual | Automatic | Semi-automatic | Automatic for customers |
| **Scale to zero** | Manual | Yes | No (real-time) / Yes (async) | Depends on customer config |
| **Auto-scaling** | Manual | Built-in | Config required | Built-in |
| **Outbound network** | Yes | Yes | Yes | **No** |
| **Summary generation** | All modes | Mode 1+2 | Mode 1+2 (custom) | Mode 2 only |
| **Customer billing** | You handle | Baseten handles | You handle | AWS handles |
| **Discovery** | None | None | None | AWS Marketplace search |
| **Monitoring** | DIY | Baseten dashboard | CloudWatch | CloudWatch |
| **Existing code reuse** | 100% | truss/ dir | 95% (add 2 routes) | 90% (bundle weights) |

---

## 12. FAQ

### Can we really do what TabPFN did?

**Yes.** TabPFN listed on SageMaker Marketplace as a subscribable model product.
Migas-1.5 has everything needed:
- A Docker container with GPU support (CUDA 12.4)
- A REST API that's almost SageMaker-compatible (2 route aliases needed)
- Model weights on HuggingFace Hub that can be bundled into the container
- A well-defined JSON request/response format

The technical gap is minimal. The Marketplace listing is more of a business/admin
process than an engineering challenge.

### How is this different from Baseten?

Baseten is a managed platform where you `truss push` and get an endpoint. SageMaker
gives you more control but requires more setup. The key differences:
- SageMaker runs in the customer's AWS account (data never leaves their environment)
- SageMaker Marketplace gives discovery and AWS-native billing
- Baseten is simpler to set up and has built-in scale-to-zero

Both can coexist — offer Baseten for quick-start users and SageMaker for
enterprise AWS customers.

### What about the vLLM dependency?

For SageMaker, the recommended approach is Mode 2 (pre-computed summaries). Users
generate summaries client-side using `generate_summary()` from the Python package,
then pass them in the request. This avoids deploying a second GPU endpoint for vLLM.

If Mode 1 is needed on SageMaker, you can deploy vLLM as a separate SageMaker
endpoint and configure Migas to point at it. This requires VPC configuration for
endpoint-to-endpoint communication.

### What if the model takes too long to start?

SageMaker gives 8 minutes for the container to pass health checks. Migas typically
loads in ~60-120 seconds on a T4 GPU. If it ever takes longer:
- The `/ping` endpoint can return 503 while loading (SageMaker will keep retrying)
- Chronos-2 download from HuggingFace is the slowest part (~30-60s first time,
  cached after)
- For Marketplace, weights are pre-bundled so startup is faster

### How do we handle authentication?

SageMaker endpoints are not publicly accessible. Callers must have valid AWS
credentials with `sagemaker:InvokeEndpoint` permission. This is handled by IAM —
no API key management needed. The existing `MIGAS_API_KEY` middleware is unnecessary
on SageMaker but doesn't hurt if left in.

### Can we use async or batch inference?

Yes. SageMaker Async Inference is a good fit for batch processing:
- Accepts large payloads via S3
- Scales to zero when idle (no charges)
- Results written to S3
- Can process multiple series in parallel

The existing `/predict` endpoint works as-is for async inference — SageMaker
handles the S3 upload/download transparently.

### What regions should we support?

GPU instances (`ml.g4dn.*`, `ml.g5.*`) are available in most major regions:
us-east-1, us-west-2, eu-west-1, ap-northeast-1. Start with **us-east-1**
(cheapest, most available) and expand based on customer demand.

### What does it cost us (not the customer)?

- **ECR storage**: ~$0.10/GB/month (image is ~2-3 GB = ~$0.30/month)
- **SageMaker endpoint**: Only if you run one for testing (~$0.74/hr for g4dn.xlarge)
- **Marketplace listing**: Free to list; AWS takes a percentage of revenue

The customer pays for the compute when they deploy an endpoint.
