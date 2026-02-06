#!/bin/bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

# ============ CONFIGURATION ============
CONFIG="config_merged_cur.yaml"
METRIC="mean_mae"
# EXCLUDE="egg_prices_with_text"  # comma-separated, e.g. "egg_prices_with_text,OIL_fred_with_text"
EXCLUDE=""
# =======================================

uv run generate_table.py \
    --config "$CONFIG" \
    --output results_table.tex \
    --standalone \
    --plot \
    --metric "$METRIC" \
    --exclude "$EXCLUDE" 

# Compile PDF
if command -v pdflatex &> /dev/null; then
    pdflatex -interaction=nonstopmode results_table_standalone.tex > /dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode results_table_standalone.tex > /dev/null 2>&1 || true
    rm -f results_table_standalone.aux results_table_standalone.log 2>/dev/null || true
    mv -f results_table_standalone.pdf results_table.pdf 2>/dev/null || true
    echo "PDF: results_table.pdf"
fi
