#!/usr/bin/env bash
# Start vLLM server for Migas-1.5 evaluation (context summarization) and --eval_gpt_forecast.
# Run this in a separate terminal before running evaluation with --eval_migas15.
# vLLM is optional in this project. Install it from repo root with:
#   uv sync --extra vllm

set -e

if ! uv run python -c "import vllm" >/dev/null 2>&1; then
  echo "vLLM is not installed in this environment."
  echo "Install the optional dependency with:"
  echo "  uv sync --extra vllm"
  exit 1
fi

# Configuration — edit as needed
MODEL="${VLLM_MODEL:-openai/gpt-oss-120b}"
PORT="${VLLM_PORT:-8004}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
# GPU memory fraction (0.9 can fail when free < 90% of total; 0.85 is safer)
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.60}"
# Number of GPUs to use for tensor parallelism (default: all in CUDA_VISIBLE_DEVICES)
# Large models (e.g. 120B) must be sharded across multiple GPUs or they OOM on one.
if [[ -n "${VLLM_TENSOR_PARALLEL_SIZE:-}" ]]; then
	  TENSOR_PARALLEL_SIZE="$VLLM_TENSOR_PARALLEL_SIZE"
  else
	    TENSOR_PARALLEL_SIZE=$(echo "$GPU" | tr ',' '\n' | wc -l)
fi

echo "Starting vLLM server for Migas-1.5 eval:"
echo "  Model: $MODEL"
echo "  Port:  $PORT"
echo "  GPU:   $GPU"
echo "  Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "  GPU memory utilization: $GPU_MEMORY_UTILIZATION"
echo "  URL:   http://localhost:$PORT/v1"
echo ""

export CUDA_VISIBLE_DEVICES="$GPU"
exec uv run vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  "$@"