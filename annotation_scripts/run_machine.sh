#!/usr/bin/env bash
# Run annotation workload for one machine in a distributed setup.
# Usage: bash annotation_scripts/run_machine.sh <MACHINE_ID>
#   MACHINE_ID: 0-based index (0–9 for 10 machines)
#
# Each machine starts its own vLLM server across all 8 GPUs, then runs
# all 3 annotation scripts with deterministic sharding so every series
# is covered exactly once across all machines.

set -e

MACHINE_ID=${1:?Usage: run_machine.sh <MACHINE_ID>  (0-based index)}
NUM_MACHINES=${NUM_MACHINES:-10}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=============================================="
echo "Machine ${MACHINE_ID} / ${NUM_MACHINES}"
echo "=============================================="

# --- Start vLLM on all 8 GPUs (dedicated annotation machines) ---
echo "Starting vLLM server on all GPUs..."
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash "$REPO_DIR/start_vllm.sh" &
VLLM_PID=$!

echo "Waiting for vLLM to become ready on port ${VLLM_PORT:-8004}..."
until curl -s "http://localhost:${VLLM_PORT:-8004}/v1/models" > /dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM process died. Check logs."
        exit 1
    fi
    sleep 5
done
echo "vLLM is ready."

# --- Run annotation scripts sequentially, each with this machine's shard ---

echo ""
echo "=== Beijing Air Quality ==="
uv run python "$SCRIPT_DIR/annotate_beijing_air_vllm.py" \
    --shard_index "$MACHINE_ID" --num_shards "$NUM_MACHINES" \
    --num_series 120

echo ""
echo "=== ETT Small ==="
uv run python "$SCRIPT_DIR/annotate_ett_vllm.py" \
    --shard_index "$MACHINE_ID" --num_shards "$NUM_MACHINES" \
    --num_series 26

echo ""
echo "=== USECON ==="
uv run python "$SCRIPT_DIR/annotate_series_vllm.py" \
    --shard_index "$MACHINE_ID" --num_shards "$NUM_MACHINES" \
    --num_series 12689

echo ""
echo "=============================================="
echo "Machine ${MACHINE_ID} finished all annotations."
echo "=============================================="
