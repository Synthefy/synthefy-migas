# Migas-1.5: Text-and-Time-Series Fusion Model

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Model-FFD21E)](https://huggingface.co/Synthefy/migas-1.5/tree/main) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HF-FNSPID%20(prepared)-FFD21E)](https://huggingface.co/datasets/Synthefy/fnspid) [![Paper](https://img.shields.io/badge/Paper-coming%20soon-1a1a2e)](https://arxiv.org/abs/)


Migas-1.5 fuses historical time series with per-step text context: an LLM summarizes the context into factual and predictive signals, and a small fusion head combines these with the univariate forecast to output the final prediction. You can use this repo in two ways:

1. **Inference with Migas-1.5** — Load the pipeline from the Hugging Face Hub and run forecasts from a DataFrame (`predict_from_dataframe()`) or low-level arrays (`predict()`).
2. **Run evaluations** — Evaluate Migas-1.5 and baselines (Chronos-2, TimesFM, Prophet, naive, etc.) on CSV/Parquet datasets via the CLI; compute MSE, MAE, MAPE, and directional accuracy.

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

If you want to run the notebook and evaluation examples in this README as-is, also download the prepared FNSPID assets (CSVs and optional summaries) from [FNSPID evaluation data](#fnspid-evaluation-data).

---

## Usage

### 1. Inference with Migas-1.5

Load the pipeline from the Hugging Face Hub and run predictions.

```python
from migaseval import MigasPipeline
import pandas as pd

pipeline = MigasPipeline.from_pretrained(
    "Synthefy/migas-1.5",
    device="cuda",
)

# DataFrame columns: t, y_t, text
df = pd.read_csv("./data/fnspid_prepared/fnspid_0.5_complement_csvs/adbe_with_text.csv")

# Returns a 1-D numpy array, shape (pred_len,)
forecast = pipeline.predict_from_dataframe(df, pred_len=16, seq_len=384)
```

For full inference with summarization, a vLLM server must be running (see [Migas-1.5 and the LLM server](#migas-15-and-the-llm-server)). The default URL and model can be overridden with `VLLM_BASE_URL` and `VLLM_MODEL`.

**Using pre-computed summaries (no LLM server needed):**

If you have pre-computed summaries (e.g. from the prepared FNSPID assets), pass them to `predict_from_dataframe()` (or `predict()`) to skip LLM generation entirely:

```python
summaries = ["FACTUAL SUMMARY: ... PREDICTIVE SIGNALS: ..."]  # one string per sample
forecast = pipeline.predict_from_dataframe(
    df, pred_len=16, seq_len=384, summaries=summaries
)
```

**Low-level API (advanced):** If you already have batched arrays, call `predict(context, text, pred_len=...)` directly where `context` has shape `(B, T)` and `text` is a list of `B` lists of strings.

For more inference scenarios (quick start, bring your own data, batch inference, backtesting), see [Example notebooks](#example-notebooks).

### 2. Running evaluations

Run the evaluation CLI on your time-series data. Each data file (CSV or Parquet) must have columns `t`, `y_t`, and `text` (see [Data format](#data-format)).

**Data source:** Put CSV or Parquet files in a folder and pass `--datasets_dir /path/to/folder`. The default is `./data/test` (or set `MIGAS_EVAL_DATASETS_DIR`). For FNSPID, download ready-to-use assets from Hugging Face — see [FNSPID evaluation data](#fnspid-evaluation-data).

**Example: baselines only (no Migas-1.5, no LLM)**

```bash
uv run python -m migaseval.evaluation \
  --datasets_dir ./data/fnspid_prepared/fnspid_0.5_complement_csvs \
  --output_dir ./results \
  --seq_len 384 \
  --pred_len 16 \
  --batch_size 64 \
  --eval_chronos2 \
  --eval_timesfm \
  --eval_prophet \
  --eval_naive
```

**Example: with Migas-1.5**

By default, `--eval_migas15` loads Migas-1.5 from the Hugging Face repo `Synthefy/migas-1.5`. You only need `--checkpoint` if you want to override that default with another HF repo id or a local checkpoint path.

```bash
uv run python -m migaseval.evaluation \
  --datasets_dir ./data/fnspid_prepared/fnspid_0.5_complement_csvs \
  --output_dir ./results \
  --seq_len 384 \
  --pred_len 16 \
  --batch_size 64 \
  --eval_migas15 \
  --eval_chronos2 \
  --eval_timesfm
```

Migas-1.5 evaluation requires a running vLLM (or OpenAI-compatible) server for context summarization. Start it before running the above (see [Migas-1.5 and the LLM server](#migas-15-and-the-llm-server)).

**Fast path: use pre-computed summaries (no LLM server needed)**

If you downloaded pre-computed summaries from Hugging Face (see [FNSPID evaluation data](#fnspid-evaluation-data)), pass `--summaries_dir` to skip on-the-fly LLM generation entirely:

```bash
uv run python -m migaseval.evaluation \
  --datasets_dir ./data/fnspid_prepared/fnspid_0.5_complement_csvs \
  --summaries_dir ./data/fnspid_prepared/fnspid_0.5_complement \
  --output_dir ./results \
  --seq_len 384 \
  --pred_len 16 \
  --batch_size 64 \
  --eval_migas15 \
  --checkpoint Synthefy/migas-1.5
```

Alternatively, you can cache your own summaries and then evaluate with the lightweight evaluator:

```bash
# Step 1) Cache summaries (requires LLM server; writes to results/<suite>/context_<seq_len>/...)
uv run python -m migaseval.evaluation \
  --datasets_dir ./data/fnspid_prepared/fnspid_0.5_complement_csvs \
  --output_dir ./results \
  --seq_len 384 \
  --pred_len 16 \
  --batch_size 64 \
  --cache_summaries

# Step 2) Evaluate from cached summaries (no LLM calls)
uv run python scripts/eval_simple.py \
  --summaries_dir ./results/output/context_384 \
  --checkpoint Synthefy/migas-1.5 \
  --context_lengths 32 64 128 256 384 \
  --eval_timesfm --eval_toto --eval_prophet
```

After running evals, generate the final post-eval report and plots:

```bash
uv run python scripts/post_eval.py --results_dir ./results/suite/context_64 --all
```

For other post-eval options (`--scatter`, `--qualitative`, or individual plotting scripts), see [Post-evaluation pipeline](#post-evaluation-pipeline).

---

## Pre-trained weights and data

| Resource   | Hugging Face |
|-----------|---------------|
| **Model** | [Synthefy/migas-1.5](https://huggingface.co/Synthefy/migas-1.5/tree/main) |
| **FNSPID (prepared)** | [Synthefy/fnspid](https://huggingface.co/datasets/Synthefy/fnspid) |

---

## FNSPID evaluation data

The [FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) dataset (Dong et al., 2024) contains 10M+ financial news articles linked to stock symbols. We provide **ready-to-use** prepared assets on Hugging Face at [Synthefy/fnspid](https://huggingface.co/datasets/Synthefy/fnspid):

- **CSVs** (`fnspid_0.5_complement_csvs/`) — files with columns `t`, `y_t`, `text`, ready for evaluation or inference.
- **Pre-computed summaries** (`fnspid_0.5_complement/`) — per-dataset subdirectories with cached LLM summaries, so you can run Migas-1.5 evaluation without a running LLM server.

### Downloading prepared assets

```bash
# Download CSVs only
uv run python scripts/download_fnspid.py --csvs

# Download pre-computed summaries only
uv run python scripts/download_fnspid.py --summaries

# Download both
uv run python scripts/download_fnspid.py --all
```

By default, assets are saved to `data/fnspid_prepared/`. Change with `--local_dir`:

```bash
uv run python scripts/download_fnspid.py --all --local_dir ./my_data
```

After downloading, use the prepared CSVs and summaries with the evaluation CLI:

```bash
uv run python -m migaseval.evaluation \
  --datasets_dir ./data/fnspid_prepared/fnspid_0.5_complement_csvs \
  --summaries_dir ./data/fnspid_prepared/fnspid_0.5_complement \
  --checkpoint Synthefy/migas-1.5 \
  --eval_migas15 --eval_chronos2 --eval_timesfm
```

<details>
<summary>Building FNSPID from raw data (advanced)</summary>

If you want to preprocess the raw FNSPID dataset yourself instead of using the prepared assets above, use `scripts/prepare_fnspid.sh` and the individual scripts under `scripts/fnspid/`. This requires downloading the raw FNSPID dataset from [Zihan1004/FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) and running an LLM server for the summarization step. See the scripts for usage details.

</details>

---

## Example notebooks

- [Migas-1.5 Inference Quick Start](notebooks/migas-inference-quickstart.ipynb) — Load the pipeline, run `predict()` on sample data, and plot the forecast.
- [Bring Your Own Data](notebooks/migas-bring-your-own-data.ipynb) — Prepare a DataFrame with `t`, `y_t`, `text` columns and forecast with `predict_from_dataframe()`.
- [Batch Inference](notebooks/migas-batch-inference.ipynb) — Forecast multiple time series in one call and iterate over a directory of CSVs.
- [Backtest and Metrics](notebooks/migas-backtest-and-metrics.ipynb) — Rolling-window backtest with MAE, MSE, MAPE, directional accuracy, and forecast-vs-ground-truth plots.

---

<!--
## Citation

If you use Migas-1.5 in your research, please cite:

```bibtex
@article{migas2025,
  title={Migas-1.5: Text-and-Time-Series Fusion Model},
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
| `--eval_migas15` | Migas-1.5 model (defaults to HF repo `Synthefy/migas-1.5`; override with `--checkpoint` or `MIGAS_CHECKPOINT`, including a local path) |
| `--eval_migas15_timesfm` | Migas-1.5 with TimesFM as univariate (requires `--checkpoint_timesfm`) |
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
          migas_pred.npy
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

### Migas-1.5 and the LLM server

Migas-1.5 and `--eval_gpt_forecast` require an LLM server for context summarization or GPT forecasts. Start vLLM (or an OpenAI-compatible server) before running. Default URL: `http://localhost:8004/v1`, default model: `openai/gpt-oss-120b`. Override with `--llm_base_url` and `--llm_model`, or `VLLM_BASE_URL` and `VLLM_MODEL`. No API key is required for a local vLLM server.

**Quick start for Migas-1.5 eval:** In a separate terminal run `bash start_vllm.sh` (or `uv run vllm serve openai/gpt-oss-120b --port 8004`). Then run the evaluation CLI with `--eval_migas15`. By default it downloads the checkpoint from `Synthefy/migas-1.5`; use `--checkpoint` only to override that source.

### Environment variables

| Variable | Meaning |
|----------|---------|
| `MIGAS_CHECKPOINT` | Override the default Migas-1.5 checkpoint source (`Synthefy/migas-1.5`) with another HF repo id or a local path |
| `MIGAS_EVAL_DATASETS_DIR` | Default `--datasets_dir` |
| `MIGAS_EVAL_OUTPUT_DIR` | Default `--output_dir` |
| `HF_TOKEN` | Hugging Face token (only needed for TabPFN) |
| `VLLM_BASE_URL` | vLLM/LLM server URL for Migas-1.5 summarizer (default `http://localhost:8004/v1`) |
| `VLLM_MODEL` | Model name for Migas-1.5 summarizer |
| `VLLM_TENSOR_PARALLEL_SIZE` | GPUs for tensor parallelism (default: number of GPUs in `CUDA_VISIBLE_DEVICES`) |

### Adding a new baseline

1. Add a module under `src/migaseval/baselines/` with a function:

   `evaluate_<name>(loader, device, pred_len, **kwargs) -> dict`

   returning `{"input": Tensor, "gt": Tensor, "predictions": {model_name: Tensor}}`.

2. In `src/migaseval/baselines/__init__.py`, import it and call:

   `register_baseline(name="<name>", eval_func=..., prediction_keys=[...], help_text="...")`.

The CLI gets a `--eval_<name>` flag automatically. Baselines that depend on another baseline’s outputs are run after their dependencies.

---

## Post-evaluation pipeline

Scripts under `scripts/` expect the same output layout as the main eval (e.g. `results/<suite>/context_64/` with `outputs/<dataset>/` and `stats_Context_64_allsamples.csv`).

- **Single entry point:** `uv run python scripts/post_eval.py --results_dir <path>` — bar plots and `report/report.md`. Add `--scatter` for scatter plots; add `--qualitative` or `--all` for qualitative forecast plots (datasets path is read from evaluation metadata when available, or pass `--datasets_dir`).
- **Bar plots only:** `scripts/plot_bars.py` — aggregate MAE, grouped bars, Migas-1.5 win rate, improvement %, ELO (if multielo installed). Usable standalone or via `post_eval.py`.
- **Scatter plots:** `scripts/plot_scatter.py` — Migas-1.5 vs baseline MAE per sample. Run standalone or with `post_eval.py --scatter`.
- **Qualitative forecast plots:** `scripts/plot_qualitative_forecasts.py` — context + ground truth + forecasts for selected samples. Run standalone or with `post_eval.py --qualitative`.

Use each script’s `--help` for full options.
