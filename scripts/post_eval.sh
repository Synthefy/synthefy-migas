#!/bin/bash
# Post-evaluation: generate bar plots and Markdown report.
# Optionally generate LaTeX table if config is provided.
#
# Usage:
#   bash scripts/post_eval.sh
#   bash scripts/post_eval.sh ./results/suite/context_64
#   bash scripts/post_eval.sh ./results/suite/context_64 scripts/latex/config_example.yaml

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RESULTS_DIR="${1:-./results/suite/context_64}"
LATEX_CONFIG="${2:-}"

if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Usage: $0 [results_dir] [latex_config.yaml]"
  echo "  results_dir: directory containing stats_Context_*_allsamples.csv (default: ./results/suite/context_64)"
  echo "  latex_config: optional path to generate_table config (e.g. scripts/latex/config_example.yaml)"
  exit 1
fi

EXTRA=()
if [[ -n "$LATEX_CONFIG" ]] && [[ -f "$LATEX_CONFIG" ]]; then
  EXTRA=(--latex_config "$LATEX_CONFIG")
fi

uv run python scripts/generate_report.py --results_dir "$RESULTS_DIR" "${EXTRA[@]}"
echo "Done. Open $(realpath "$RESULTS_DIR")/report/report.md for the report."
