#!/usr/bin/env bash
# Start vLLM server for TTFM evaluation (context summarization) and --eval_gpt_forecast.
# Run this in a separate terminal before running evaluation with --eval_ttfmlf.
# Requires vLLM in this project's environment (e.g. from repo root: uv add vllm).

set -e

# Configuration — edit as needed
MODEL="${VLLM_MODEL:-openai/gpt-oss-120b}"
PORT="${VLLM_PORT:-8004}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
# Number of GPUs to use for tensor parallelism (default: all in CUDA_VISIBLE_DEVICES)
# Large models (e.g. 120B) must be sharded across multiple GPUs or they OOM on one.
if [[ -n "${VLLM_TENSOR_PARALLEL_SIZE:-}" ]]; then
  TENSOR_PARALLEL_SIZE="$VLLM_TENSOR_PARALLEL_SIZE"
else
  TENSOR_PARALLEL_SIZE=$(echo "$GPU" | tr ',' '\n' | wc -l)
fi

echo "Starting vLLM server for TTFM eval:"
echo "  Model: $MODEL"
echo "  Port:  $PORT"
echo "  GPU:   $GPU"
echo "  Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "  URL:   http://localhost:$PORT/v1"
echo ""

export CUDA_VISIBLE_DEVICES="$GPU"
exec uv run vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  "$@"
