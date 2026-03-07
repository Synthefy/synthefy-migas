#!/usr/bin/env bash
# Example: run TTFM + Chronos-2 + TimesFM. Adjust CUDA devices and paths as needed.

export CUDA_VISIBLE_DEVICES=0,1,2

# Optional: set defaults via env (override with CLI)
# export TTFM_CHECKPOINT=Synthefy/ttfm
# export TTFM_EVAL_DATASETS_DIR=./data/test
# export TTFM_EVAL_OUTPUT_DIR=./results

uv run python -m ttfmeval.evaluation \
  --seq_len 64 \
  --pred_len 16 \
  --batch_size 64 \
  --datasets_dir "${TTFM_EVAL_DATASETS_DIR:-./data/test}" \
  --output_dir "${TTFM_EVAL_OUTPUT_DIR:-./results}" \
  --device cuda \
  --eval_chronos2 \
  --eval_timesfm \
  "$@"

# To also run TTFM, add checkpoint (HF repo id, e.g. Synthefy/ttfm; set HF_TOKEN for private):
#   --eval_ttfmlf \
#   --checkpoint "${TTFM_CHECKPOINT:-Synthefy/ttfm}"
