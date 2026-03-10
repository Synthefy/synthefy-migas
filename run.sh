#!/usr/bin/env bash
# Example: run Migas-1.5 + baselines. Adjust CUDA devices and paths as needed.

export CUDA_VISIBLE_DEVICES=0,1,2

# Optional: set defaults via env (override with CLI)
# export MIGAS_CHECKPOINT=Synthefy/migas-1.5
# export MIGAS_EVAL_DATASETS_DIR=./data/test
# export MIGAS_EVAL_OUTPUT_DIR=./results

uv run python scripts/eval_simple.py \
  --datasets_dir "${MIGAS_EVAL_DATASETS_DIR:-./data/test}" \
  --output_dir "${MIGAS_EVAL_OUTPUT_DIR:-./results}" \
  --seq_len 64 \
  --pred_len 16 \
  --batch_size 64 \
  --device cuda \
  --eval_timesfm \
  "$@"

# To also run Migas-1.5 (requires LLM server for first run):
#   --eval_migas15 \
#   --checkpoint "${MIGAS_CHECKPOINT:-Synthefy/migas-1.5}"
