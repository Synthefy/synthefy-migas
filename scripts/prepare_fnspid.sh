#!/usr/bin/env bash
#
# Prepare FNSPID evaluation data for TTFM.
#
# Downloads the FNSPID dataset (Zihan1004/FNSPID) which includes both news
# articles and stock price history. Extracts per-symbol text and (optionally)
# creates LLM daily summaries + merges into the final CSV format (t, y_t, text).
#
# Usage:
#   bash scripts/prepare_fnspid.sh                         # default symbols (ADBE,JPM,NFLX,AMD)
#   bash scripts/prepare_fnspid.sh --all                   # all 100 high-annotation symbols
#   bash scripts/prepare_fnspid.sh --symbols AAPL,MSFT,BA  # specific symbols
#   bash scripts/prepare_fnspid.sh --top-k 20              # top 20 by text availability
#   bash scripts/prepare_fnspid.sh --skip-summaries        # stop before LLM step
#
# The final CSVs land in data/fnspid/output/ and can be used directly with
# the evaluation CLI:
#   uv run python -m ttfmeval.evaluation --datasets_dir data/fnspid/output ...

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/scripts/fnspid"
WORK_DIR="${REPO_ROOT}/data/fnspid"

RAW_DIR="${WORK_DIR}/raw"
HISTORY_DIR="${WORK_DIR}/full_history"
EXTRACTED_DIR="${WORK_DIR}/extracted_text"
SUMMARIES_DIR="${WORK_DIR}/summaries"
OUTPUT_DIR="${WORK_DIR}/output"

# Defaults
SYMBOLS=""
TOP_K=""
SKIP_SUMMARIES=false
LLM_URL="http://localhost:8004/v1"
LLM_MODEL="openai/gpt-oss-120b"

# All 100 high-annotation symbols from FNSPID
ALL_SYMBOLS="AAL,ABBV,ABT,ADBE,AIR,ALB,AMAT,AMC,AMD,AMGN,ANTM,BABA,BHP,BIDU,BIIB,BKR,BLK,BNTX,BX,CAT,CMCSA,CMG,COP,COST,CRM,CRWD,CVX,DGAZ,DIS,DKNG,DRR,ENB,ENPH,FCAU,FCX,FDX,GE,GILD,GLD,GME,GOOGL,GOOG,GSK,GS,GWPH,HYG,INTC,JD,JPM,KKR,KO,KR,MDT,MMM,MRK,MSFT,MS,MU,MYL,NEE,NFLX,NIO,NKE,NVAX,ORCL,OXY,PEP,PINS,PLUG,PM,PTON,QCOM,QQQ,SBUX,SIRI,SLB,SO,SPOT,SPY,STZ,TDOC,TMUS,TSCO,TSM,T,TXN,UGAZ,UPS,USO,VRTX,V,WBA,WFC,WMT,XLF,XLK,XLP,XLY,XOM,ZM"
DEFAULT_SYMBOLS="ADBE,JPM,NFLX,AMD"

usage() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --symbols SYM1,SYM2,...   Comma-separated stock symbols (default: ${DEFAULT_SYMBOLS})"
  echo "  --all                     Process all 100 high-annotation symbols"
  echo "  --top-k N                 Instead of --symbols, use top N from metadata.csv by text availability"
  echo "  --skip-summaries          Stop before the LLM daily-summary step (steps 1-3 only)"
  echo "  --llm-base-url URL        LLM server base URL (default: ${LLM_URL})"
  echo "  --llm-model MODEL         LLM model name (default: ${LLM_MODEL})"
  echo "  --work-dir DIR            Working directory (default: ${WORK_DIR})"
  echo "  -h, --help                Show this help"
  echo ""
  echo "All 100 high-annotation symbols:"
  echo "  ${ALL_SYMBOLS}"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --symbols) SYMBOLS="$2"; shift 2 ;;
    --all) SYMBOLS="$ALL_SYMBOLS"; shift ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --skip-summaries) SKIP_SUMMARIES=true; shift ;;
    --llm-base-url) LLM_URL="$2"; shift 2 ;;
    --llm-model) LLM_MODEL="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

if [[ -z "$SYMBOLS" && -z "$TOP_K" ]]; then
  SYMBOLS="$DEFAULT_SYMBOLS"
fi

mkdir -p "$WORK_DIR"

# ── Step 1: Download FNSPID dataset from Hugging Face ────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "Step 1: Download FNSPID dataset from Hugging Face (Zihan1004/FNSPID)"
echo "═══════════════════════════════════════════════════════════════════"

if [[ -d "$RAW_DIR/Stock_news" ]] && [[ -d "$RAW_DIR/Stock_price" ]]; then
  echo "Raw data already exists in ${RAW_DIR}, skipping download."
else
  echo "Downloading to ${RAW_DIR}..."
  huggingface-cli download Zihan1004/FNSPID \
    --repo-type dataset \
    --local-dir "$RAW_DIR"
  echo "Download complete."
fi

# The FNSPID repo structure:
#   Stock_news/All_external.csv          (~5.7 GB, articles with Article body)
#   Stock_news/nasdaq_exteral_data.csv   (~23 GB, articles with more columns)
#   Stock_price/full_history.zip         (~590 MB, per-symbol price CSVs)

# Find the news CSV (prefer nasdaq_exteral_data.csv as it has richer data)
FNSPID_CSV=""
for candidate in \
  "$RAW_DIR/Stock_news/nasdaq_exteral_data.csv" \
  "$RAW_DIR/Stock_news/All_external.csv"; do
  if [[ -f "$candidate" ]]; then
    FNSPID_CSV="$candidate"
    break
  fi
done

if [[ -z "$FNSPID_CSV" ]]; then
  # Fallback: search recursively
  FNSPID_CSV=$(find "$RAW_DIR" -name '*.csv' -type f | head -1)
fi

if [[ -z "$FNSPID_CSV" ]]; then
  echo "Error: No CSV file found in ${RAW_DIR}."
  echo "Contents:"
  find "$RAW_DIR" -type f | head -20
  exit 1
fi
echo "Using news CSV: ${FNSPID_CSV}"

# ── Step 2: Extract price history from bundled zip ───────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "Step 2: Extract stock price history from FNSPID bundle"
echo "═══════════════════════════════════════════════════════════════════"

HISTORY_ZIP="$RAW_DIR/Stock_price/full_history.zip"

if [[ -d "$HISTORY_DIR" ]] && ls "$HISTORY_DIR"/*.csv &>/dev/null 2>&1; then
  echo "Price history already extracted in ${HISTORY_DIR}, skipping."
else
  if [[ -f "$HISTORY_ZIP" ]]; then
    echo "Unzipping ${HISTORY_ZIP} -> ${HISTORY_DIR}/ ..."
    mkdir -p "$HISTORY_DIR"
    unzip -o -q "$HISTORY_ZIP" -d "$HISTORY_DIR"
    # If the zip created a nested directory, flatten it
    if [[ -d "$HISTORY_DIR/full_history" ]]; then
      mv "$HISTORY_DIR/full_history"/* "$HISTORY_DIR"/ 2>/dev/null || true
      rmdir "$HISTORY_DIR/full_history" 2>/dev/null || true
    fi
    CSV_COUNT=$(ls "$HISTORY_DIR"/*.csv 2>/dev/null | wc -l)
    echo "Extracted ${CSV_COUNT} price history CSVs."
  else
    echo "Warning: No full_history.zip found at ${HISTORY_ZIP}."
    echo "Price CSVs can be downloaded separately with:"
    echo "  uv run python scripts/fnspid/download_prices.py AAPL MSFT ... --output-dir ${HISTORY_DIR}"
    echo "Continuing without price data (only text JSONs will be created)."
  fi
fi

# ── Step 3: Extract text per symbol ──────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "Step 3: Extract text per symbol from FNSPID"
echo "═══════════════════════════════════════════════════════════════════"

HISTORY_ARG=""
if [[ -d "$HISTORY_DIR" ]] && ls "$HISTORY_DIR"/*.csv &>/dev/null 2>&1; then
  HISTORY_ARG="--history-dir $HISTORY_DIR"
fi

if [[ -n "$SYMBOLS" ]]; then
  SYMBOL_ARGS=$(echo "$SYMBOLS" | tr ',' ' ')
  uv run python "${SCRIPT_DIR}/extract_text.py" \
    $SYMBOL_ARGS \
    --input "$FNSPID_CSV" \
    --output-dir "$EXTRACTED_DIR" \
    $HISTORY_ARG
else
  echo "No symbols specified yet (using --top-k). Extracting all symbols..."
  UNIQUE_SYMBOLS=$(uv run python -c "
import polars as pl
df = pl.read_csv('$FNSPID_CSV', columns=['Stock_symbol'])
syms = df['Stock_symbol'].unique().sort().to_list()
print(' '.join(s for s in syms if s and s.strip()))
")
  uv run python "${SCRIPT_DIR}/extract_text.py" \
    $UNIQUE_SYMBOLS \
    --input "$FNSPID_CSV" \
    --output-dir "$EXTRACTED_DIR" \
    $HISTORY_ARG
fi

# ── Step 3b: Analyse text availability ───────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "Step 3b: Analyse text availability (metadata.csv)"
echo "═══════════════════════════════════════════════════════════════════"

METADATA_CSV="${WORK_DIR}/metadata.csv"
uv run python "${SCRIPT_DIR}/analyse_text.py" \
  --extracted-dir "$EXTRACTED_DIR" \
  --output "$METADATA_CSV"

if $SKIP_SUMMARIES; then
  echo ""
  echo "═══════════════════════════════════════════════════════════════════"
  echo "Done (--skip-summaries). Extracted text and metadata are in:"
  echo "  ${EXTRACTED_DIR}/"
  echo "  ${METADATA_CSV}"
  echo ""
  echo "To continue with LLM summaries, re-run without --skip-summaries"
  echo "and ensure an LLM server is running at ${LLM_URL}."
  echo "═══════════════════════════════════════════════════════════════════"
  exit 0
fi

# ── Step 4: Create daily summaries + merge ───────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "Step 4: Create daily summaries (LLM) + merge into final CSVs"
echo "═══════════════════════════════════════════════════════════════════"
echo "LLM server: ${LLM_URL}"
echo "LLM model:  ${LLM_MODEL}"

if [[ -n "$TOP_K" ]]; then
  uv run python "${SCRIPT_DIR}/run_top_k.py" \
    --metadata "$METADATA_CSV" \
    --top-k "$TOP_K" \
    --extracted-dir "$EXTRACTED_DIR" \
    --summaries-dir "$SUMMARIES_DIR" \
    --data-dir "$OUTPUT_DIR" \
    --llm-base-url "$LLM_URL" \
    --llm-model "$LLM_MODEL"
else
  for SYMBOL in $(echo "$SYMBOLS" | tr ',' ' '); do
    SYM_LOWER=$(echo "$SYMBOL" | tr '[:upper:]' '[:lower:]')
    TEXT_JSON="${EXTRACTED_DIR}/${SYM_LOWER}_text.json"
    NUM_CSV="${EXTRACTED_DIR}/${SYM_LOWER}.csv"

    if [[ ! -f "$TEXT_JSON" ]]; then
      echo "Skip ${SYMBOL}: no text JSON"
      continue
    fi
    if [[ ! -f "$NUM_CSV" ]]; then
      echo "Skip ${SYMBOL}: no price CSV"
      continue
    fi

    echo ""
    echo "Processing ${SYMBOL}..."
    mkdir -p "$SUMMARIES_DIR" "$OUTPUT_DIR"

    uv run python "${SCRIPT_DIR}/create_daily_summaries.py" \
      --input "$TEXT_JSON" \
      --output "${SUMMARIES_DIR}/${SYM_LOWER}_text_daily.json" \
      --dates-csv "$NUM_CSV" \
      --llm-base-url "$LLM_URL" \
      --llm-model "$LLM_MODEL"

    uv run python "${SCRIPT_DIR}/merge_text_numerical.py" \
      --summaries "${SUMMARIES_DIR}/${SYM_LOWER}_text_daily.json" \
      --numerical "$NUM_CSV" \
      --output "${OUTPUT_DIR}/${SYM_LOWER}_with_text.csv"
  done
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "Done! Final CSVs (t, y_t, text) are in: ${OUTPUT_DIR}/"
echo ""
echo "Run evaluation with:"
echo "  uv run python -m ttfmeval.evaluation --datasets_dir ${OUTPUT_DIR} ..."
echo "═══════════════════════════════════════════════════════════════════"
