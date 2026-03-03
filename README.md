# TTFM: Text-and-Time-Series Fusion Model

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Model-FFD21E)](https://huggingface.co/bekzatajan/ttfm/tree/main) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HF-FNSPID%20Dataset-FFD21E)](https://huggingface.co/datasets/Zihan1004/FNSPID) [![Paper](https://img.shields.io/badge/Paper-coming%20soon-1a1a2e)](https://arxiv.org/abs/)


TTFM fuses historical time series with per-step text context: an LLM summarizes the context into factual and predictive signals, and a small fusion head combines these with the univariate forecast to output the final prediction. You can use this repo in two ways:

1. **Run evaluations** — Evaluate TTFM and baselines (Chronos-2, TimesFM, Prophet, naive, etc.) on CSV/Parquet datasets via the CLI; compute MSE, MAE, MAPE, and directional accuracy.
2. **Inference with TTFM** — Load the pipeline from the Hugging Face Hub and run `predict()` on your context and text in Python or notebooks.

---

## Installation

**Prerequisites:** Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (a fast Python package manager):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from the repo root:

```bash
uv sync
# or: pip install -e .
```
---

## Usage

### 1. Running evaluations

Run the evaluation CLI on your time-series data. Each data file (CSV or Parquet) must have columns `t`, `y_t`, and `text` (see [Data format](#data-format)).

**Data source:** Put CSV or Parquet files in a folder and pass `--datasets_dir /path/to/folder`. The default is `./data/test` (or set `TTFM_EVAL_DATASETS_DIR`). To prepare FNSPID data, see [Preparing FNSPID evaluation data](#preparing-fnspid-evaluation-data).

**Example: baselines only (no TTFM, no LLM)**

```bash
uv run python -m ttfmeval.evaluation \
  --datasets_dir ./data/fnspid/output \
  --output_dir ./results \
  --seq_len 384 \
  --pred_len 16 \
  --batch_size 64 \
  --eval_chronos2 \
  --eval_timesfm \
  --eval_prophet \
  --eval_naive
```

**Example: with TTFM**

Evaluating TTFM requires the TTFM checkpoint from the Hub: pass `--checkpoint bekzatajan/ttfm`. 

```bash
uv run python -m ttfmeval.evaluation \
  --datasets_dir ./data/fnspid/output \
  --output_dir ./results \
  --seq_len 384 \
  --pred_len 16 \
  --batch_size 64 \
  --eval_ttfmlf \
  --eval_chronos2 \
  --eval_timesfm \
  --checkpoint bekzatajan/ttfm
```

TTFM evaluation requires a running vLLM (or OpenAI-compatible) server for context summarization. Start it before running the above (see [TTFM and the LLM server](#ttfm-and-the-llm-server)).

### 2. Inference with TTFM

Load the pipeline from the Hugging Face Hub and run predictions.

```python
from ttfmeval import TTFMPipeline

pipeline = TTFMPipeline.from_pretrained(
    "bekzatajan/ttfm",
    device="cuda",
)

# context: (batch, time_steps), text: list of list of strings (one per timestep per sample)
import numpy as np
context = np.random.randn(2, 64).astype(np.float32)  # 2 samples, 64 steps
text = [["text at step %d" % i for i in range(64)] for _ in range(2)]  # 2 x 64 strings

forecast = pipeline.predict(context, text, pred_len=16)  # (2, 16, 1)
```

For full inference with summarization, a vLLM server must be running (see [TTFM and the LLM server](#ttfm-and-the-llm-server)). The default URL and model can be overridden with `VLLM_BASE_URL` and `VLLM_MODEL`.

---

## Pre-trained weights

| Resource   | Hugging Face |
|-----------|---------------|
| **Model** | [bekzatajan/ttfm](https://huggingface.co/bekzatajan/ttfm/tree/main) |

---

## Preparing FNSPID evaluation data

The [FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) dataset (Dong et al., 2024) contains 10M+ financial news articles linked to stock symbols. To use it for TTFM evaluation, the raw articles must be preprocessed into CSVs with columns `t`, `y_t`, `text`.

A single shell script handles the full pipeline:

```bash
# Default: 4 symbols (ADBE, JPM, NFLX, AMD)
bash scripts/prepare_fnspid.sh

# All 100 high-annotation symbols
bash scripts/prepare_fnspid.sh --all
```

The script supports 100 high-annotation symbols from FNSPID: AAL, ABBV, ABT, ADBE, AIR, ALB, AMAT, AMC, AMD, AMGN, ANTM, BABA, BHP, BIDU, BIIB, BKR, BLK, BNTX, BX, CAT, CMCSA, CMG, COP, COST, CRM, CRWD, CVX, DGAZ, DIS, DKNG, DRR, ENB, ENPH, FCAU, FCX, FDX, GE, GILD, GLD, GME, GOOGL, GOOG, GSK, GS, GWPH, HYG, INTC, JD, JPM, KKR, KO, KR, MDT, MMM, MRK, MSFT, MS, MU, MYL, NEE, NFLX, NIO, NKE, NVAX, ORCL, OXY, PEP, PINS, PLUG, PM, PTON, QCOM, QQQ, SBUX, SIRI, SLB, SO, SPOT, SPY, STZ, TDOC, TMUS, TSCO, TSM, T, TXN, UGAZ, UPS, USO, VRTX, V, WBA, WFC, WMT, XLF, XLK, XLP, XLY, XOM, ZM.

It runs five stages:

1. **Download** the FNSPID dataset from Hugging Face (`Zihan1004/FNSPID`), including bundled price history
2. **Extract** stock price history from the bundled zip
3. **Extract** per-symbol article text from the raw CSV
4. **Analyse** text availability and produce `metadata.csv`
5. **Summarize** articles into daily summaries using an LLM and merge with price data

Steps 1-4 are fully automatic. Step 5 requires a running LLM server (see [TTFM and the LLM server](#ttfm-and-the-llm-server)).

**Options:**

```bash
# Specific symbols
bash scripts/prepare_fnspid.sh --symbols AAPL,MSFT,BA

# All 100 high-annotation symbols
bash scripts/prepare_fnspid.sh --all

# Top 20 symbols by text availability (from metadata.csv)
bash scripts/prepare_fnspid.sh --top-k 20

# Stop before the LLM step (steps 1-4 only)
bash scripts/prepare_fnspid.sh --skip-summaries

# Custom LLM server
bash scripts/prepare_fnspid.sh --llm-base-url http://localhost:8004/v1 --llm-model openai/gpt-oss-120b
```

The final CSVs land in `data/fnspid/output/` and can be passed directly to the evaluation CLI:

```bash
uv run python -m ttfmeval.evaluation \
  --datasets_dir data/fnspid/output \
  --checkpoint bekzatajan/ttfm \
  --eval_ttfmlf --eval_chronos2 --eval_timesfm
```

<details>
<summary>Running individual preprocessing steps manually</summary>

The scripts in `scripts/fnspid/` can also be run individually:

```bash
# 1. Download FNSPID from Hugging Face
huggingface-cli download Zihan1004/FNSPID --repo-type dataset --local-dir data/fnspid/raw

# 2. Download price history (requires: pip install yfinance)
uv run python scripts/fnspid/download_prices.py AAPL MSFT --output-dir data/fnspid/full_history

# 3. Extract text per symbol
uv run python scripts/fnspid/extract_text.py AAPL MSFT \
  --input data/fnspid/raw/<fnspid_csv_file>.csv \
  --output-dir data/fnspid/extracted_text \
  --history-dir data/fnspid/full_history

# 4. Analyse text availability
uv run python scripts/fnspid/analyse_text.py \
  --extracted-dir data/fnspid/extracted_text \
  --output data/fnspid/metadata.csv

# 5a. Create daily summaries (needs LLM server)
uv run python scripts/fnspid/create_daily_summaries.py \
  --input data/fnspid/extracted_text/aapl_text.json \
  --output data/fnspid/summaries/aapl_text_daily.json \
  --dates-csv data/fnspid/extracted_text/aapl.csv

# 5b. Merge text + numerical
uv run python scripts/fnspid/merge_text_numerical.py \
  --summaries data/fnspid/summaries/aapl_text_daily.json \
  --numerical data/fnspid/extracted_text/aapl.csv \
  --output data/fnspid/output/aapl_with_text.csv

# Or batch steps 5a+5b for top K symbols:
uv run python scripts/fnspid/run_top_k.py \
  --metadata data/fnspid/metadata.csv --top-k 20 \
  --extracted-dir data/fnspid/extracted_text \
  --summaries-dir data/fnspid/summaries \
  --data-dir data/fnspid/output
```

</details>

---

## Example notebooks

- [TTFM Inference Quick Start](notebooks/ttfm-inference-quickstart.ipynb) — Load the pipeline, run `predict()` on sample data, and plot the forecast.

---

<!--
## Citation

If you use TTFM in your research, please cite:

```bibtex
@article{ttfm2025,
  title={TTFM: Text-and-Time-Series Fusion Model},
  author={...},
  journal={...},
  year={2025},
  url={...}
}
```
-->

---

## Reference

### Data format

- **Input:** A directory of CSV or Parquet files. Each file must have:
  - `t` — timestamp (e.g. `1980-01-31`)
  - `y_t` — target value (numeric)
  - `text` — per-step text context (can be empty string)

- **Supported formats:** `.csv` and `.parquet`.
- **Discovery:** All `.csv` and `.parquet` files under the given directory are used, except those whose basename ends with `_embeddings`, `_original`, `_temp`, `_results`, or `_temp_results`.

### Baselines

Enable with `--eval_<name>`:

| Flag | Description |
|------|-------------|
| `--eval_ttfmlf` | TTFM model (requires `--checkpoint` or `TTFM_CHECKPOINT`; use HF repo id e.g. bekzatajan/ttfm) |
| `--eval_ttfmlf_timesfm` | TTFM with TimesFM as univariate (requires `--checkpoint_timesfm`) |
| `--eval_chronos2` | Chronos-2 univariate |
| `--eval_chronos2_multivar` | Chronos-2 with covariates |
| `--eval_chronos2_gpt` | Chronos-2 + GPT forecasts (run `--eval_gpt_forecast` first or have cached) |
| `--eval_gpt_forecast` | LLM-only forecast (needs vLLM / OpenAI-compatible server) |
| `--eval_timesfm` | TimesFM 2.5 univariate |
| `--eval_naive` | Naive (last value) |
| `--eval_prophet` | Prophet |
| `--eval_tabpfn` | TabPFN 2.5 time-series (requires `HF_TOKEN`) |
| `--eval_toto` | Toto (optional: `pip install -e ".[toto]"`) |

### Output layout

Under `--output_dir`:

```
<output_dir>/
  <datasets_dir_name>/
    context_<seq_len>/
      eval_meta.json          # datasets_dir, seq_len, pred_len (for post_eval)
      stats_Context_<seq_len>_allsamples.csv
      stats_Context_<seq_len>_june2024plus.csv  (if applicable)
      outputs/
        <dataset_name>/
          input.npy
          gt.npy
          ttfm_pred.npy
          chronos_univar_pred.npy
          ...
```

Re-runs skip datasets that already have all requested model outputs. Use a new `--output_dir` or delete `outputs/<dataset_name>/` to re-evaluate.

### After evaluation: report and plots

```bash
uv run python scripts/post_eval.py --results_dir ./results/<suite>/context_<seq_len>
```

This generates bar plots and writes `report/report.md` in the results directory. To also include scatter plots or qualitative forecast plots:

```bash
# Bar plots + scatter plots
uv run python scripts/post_eval.py --results_dir ./results/suite/context_64 --scatter

# Bar plots + qualitative forecast plots (datasets_dir from eval metadata, or pass --datasets_dir)
uv run python scripts/post_eval.py --results_dir ./results/suite/context_64 --qualitative
uv run python scripts/post_eval.py --results_dir ./results/suite/context_64 --qualitative --datasets_dir ./data/test

# All of the above
uv run python scripts/post_eval.py --results_dir ./results/suite/context_64 --all
```

You can also run individual scripts (e.g. `scripts/plot_bars.py --results_dir ...`) for more control; see [Post-evaluation pipeline](#post-evaluation-pipeline).

### TTFM and the LLM server

TTFM and `--eval_gpt_forecast` require an LLM server for context summarization or GPT forecasts. Start vLLM (or an OpenAI-compatible server) before running. Default URL: `http://localhost:8004/v1`, default model: `openai/gpt-oss-120b`. Override with `--llm_base_url` and `--llm_model`, or `VLLM_BASE_URL` and `VLLM_MODEL`. No API key is required for a local vLLM server.

**Quick start for TTFM eval:** In a separate terminal run `bash start_vllm.sh` (or `uv run vllm serve openai/gpt-oss-120b --port 8004`). Then run the evaluation CLI with `--eval_ttfmlf` and `--checkpoint`.

### Environment variables

| Variable | Meaning |
|----------|---------|
| `TTFM_CHECKPOINT` | Default HF repo id for TTFM checkpoint (used if `--checkpoint` is not set) |
| `TTFM_EVAL_DATASETS_DIR` | Default `--datasets_dir` |
| `TTFM_EVAL_OUTPUT_DIR` | Default `--output_dir` |
| `HF_TOKEN` | Hugging Face token (only needed for TabPFN) |
| `VLLM_BASE_URL` | vLLM/LLM server URL for TTFM summarizer (default `http://localhost:8004/v1`) |
| `VLLM_MODEL` | Model name for TTFM summarizer |
| `VLLM_TENSOR_PARALLEL_SIZE` | GPUs for tensor parallelism (default: number of GPUs in `CUDA_VISIBLE_DEVICES`) |

### Adding a new baseline

1. Add a module under `src/ttfmeval/baselines/` with a function:

   `evaluate_<name>(loader, device, pred_len, **kwargs) -> dict`

   returning `{"input": Tensor, "gt": Tensor, "predictions": {model_name: Tensor}}`.

2. In `src/ttfmeval/baselines/__init__.py`, import it and call:

   `register_baseline(name="<name>", eval_func=..., prediction_keys=[...], help_text="...")`.

The CLI gets a `--eval_<name>` flag automatically. Baselines that depend on another baseline’s outputs are run after their dependencies.

---

## Post-evaluation pipeline

Scripts under `scripts/` expect the same output layout as the main eval (e.g. `results/<suite>/context_64/` with `outputs/<dataset>/` and `stats_Context_64_allsamples.csv`).

- **Single entry point:** `uv run python scripts/post_eval.py --results_dir <path>` — bar plots and `report/report.md`. Add `--scatter` for scatter plots; add `--qualitative` or `--all` for qualitative forecast plots (datasets path is read from evaluation metadata when available, or pass `--datasets_dir`).
- **Bar plots only:** `scripts/plot_bars.py` — aggregate MAE, grouped bars, TTFM win rate, improvement %, ELO (if multielo installed). Usable standalone or via `post_eval.py`.
- **Scatter plots:** `scripts/plot_scatter.py` — TTFM vs baseline MAE per sample. Run standalone or with `post_eval.py --scatter`.
- **Qualitative forecast plots:** `scripts/plot_qualitative_forecasts.py` — context + ground truth + forecasts for selected samples. Run standalone or with `post_eval.py --qualitative`.

Use each script’s `--help` for full options.
