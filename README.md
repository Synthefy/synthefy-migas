# TTFM: Text-and-Time-Series Fusion Model

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Model-FFD21E)](https://huggingface.co/bekzatajan/ttfm/tree/main) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Dataset-FFD21E)](https://huggingface.co/datasets/bekzatajan/fnspid/tree/main)


This repository provides **TTFM** (Text-and-Time-Series Fusion Model) inference and evaluation. TTFM combines a univariate time-series backbone (Chronos-2, TimesFM, or Prophet) with LLM-generated context summaries to produce text-conditioned forecasts. You can run benchmarks across TTFM and baselines on CSV datasets, or run inference in Python or notebooks.

---

## Introduction

TTFM fuses historical time series with per-step text context: an LLM summarizes the context into factual and predictive signals, and a small fusion head combines these with the univariate forecast to output the final prediction. This package supports:

- **Evaluation** — Run TTFM and baselines (Chronos-2, TimesFM, Prophet, naive, etc.) on CSV or Hugging Face datasets via the CLI; compute MSE, MAE, MAPE, and directional accuracy. Evaluating TTFM requires loading the pre-trained TTFM weights from the Hugging Face Hub.
- **Inference** — Load the pre-trained weights from the Hugging Face Hub and run forecasts in Python or notebooks.

Pre-trained weights and sample datasets are hosted on Hugging Face. !!!! Probably Sai is going to upload the model and datasets on HF, they are currently private; use `HF_TOKEN` for access. They will be made public in a future release !!!!

---

## Two ways to use this repo

1. **Run evaluations** — Evaluate TTFM and baselines on CSV (or Hugging Face) datasets using the CLI.
2. **Inference with TTFM** — Load the pipeline from the Hugging Face Hub and call `predict()` on your context and text.

---

## Installation

From the repo root:

```bash
uv sync
# or: pip install -e .
```

To install from PyPI:

```bash
pip install synthefy-ttfm
```

For TTFM evaluation or inference with context summarization, vLLM is included by default; you also need a running LLM server (see [TTFM and the LLM server](#ttfm-and-the-llm-server)).

---

## Usage

### 1. Running evaluations

Run the evaluation CLI on your time-series data. Each CSV must have columns `t`, `y_t`, and `text` (see [Data format](#data-format)).

**Data sources:** You can pass data in either way (no need to use Hugging Face):

- **Local directory (default)** — Put CSV files in a folder and pass `--datasets_dir /path/to/folder`. The default is `./data/test` (or set `TTFM_EVAL_DATASETS_DIR`).
- **Hugging Face dataset (optional)** — Use `--datasets_hf REPO_ID` (and optionally `--datasets_hf_subdir data`) to download a dataset repo from the Hub and use it as the data source. Useful for shared or published datasets.

**Example: baselines only (no TTFM, no LLM)**

```bash
uv run python -m ttfmeval.evaluation \
  --datasets_dir ./data/test \
  --output_dir ./results \
  --seq_len 384 \
  --pred_len 16 \
  --batch_size 64 \
  --eval_chronos2 \
  --eval_timesfm \
  --eval_prophet \
  --eval_naive
```

**Example: with TTFM (local data or Hugging Face)**

Evaluating TTFM requires the TTFM checkpoint from the Hub: pass `--checkpoint bekzatajan/ttfm` (for private repos, set `HF_TOKEN`). You can use the same local `--datasets_dir` as above, or switch to a Hugging Face dataset with `--datasets_hf`.

```bash
export HF_TOKEN=your_token   # only for private model/dataset repos

uv run python -m ttfmeval.evaluation \
  --datasets_dir ./data/test \
  --output_dir ./results \
  --seq_len 384 \
  --pred_len 16 \
  --batch_size 64 \
  --eval_ttfmlf \
  --eval_chronos2 \
  --eval_timesfm \
  --checkpoint bekzatajan/ttfm
# Or use a Hugging Face dataset: add --datasets_hf bekzatajan/fnspid --datasets_hf_subdir data (and omit or override --datasets_dir)
```

TTFM evaluation requires a running vLLM (or OpenAI-compatible) server for context summarization. Start it before running the above (see [TTFM and the LLM server](#ttfm-and-the-llm-server)).

### 2. Inference with TTFM

Load the pipeline from the Hugging Face Hub and run predictions.

```python
import os
from ttfmeval import TTFMPipeline

# Load from Hugging Face (set HF_TOKEN for private repos)
pipeline = TTFMPipeline.from_pretrained(
    "bekzatajan/ttfm",
    token=os.environ.get("HF_TOKEN"),
    device="cuda",
)

# context: (batch, time_steps), text: list of list of strings (one per timestep per sample)
import numpy as np
context = np.random.randn(2, 64).astype(np.float32)  # 2 samples, 64 steps
text = [["report text at t=1", "report text at t=2", ...] for _ in range(2)]  # 2 x 64 strings

forecast = pipeline.predict(context, text, pred_len=16)  # (2, 16, 1)
```

For full inference with summarization, a vLLM server must be running and `VLLM_BASE_URL` / `VLLM_MODEL` must be set (see [TTFM and the LLM server](#ttfm-and-the-llm-server)).

---

## Pre-trained weights and sample datasets

| Resource   | Hugging Face |
|-----------|---------------|
| **Model** | [bekzatajan/ttfm](https://huggingface.co/bekzatajan/ttfm/tree/main) |
| **Sample datasets** | [bekzatajan/fnspid](https://huggingface.co/datasets/bekzatajan/fnspid/tree/main) — CSVs in `data/`; use `--datasets_hf bekzatajan/fnspid --datasets_hf_subdir data` in the CLI. |

Both are currently private. Set the `HF_TOKEN` environment variable (or pass `token=` where supported) to access them. They will be made public in a future release.

### Using a Hugging Face dataset (optional)

You only need this if you want to use or publish datasets on the Hub. For local evaluation, a directory of CSVs and `--datasets_dir` is enough.

The current example dataset is [bekzatajan/fnspid](https://huggingface.co/datasets/bekzatajan/fnspid) (CSVs in the `data/` subfolder). To add or host your own:

1. **Create a new dataset repo**
   - Go to [huggingface.co/datasets](https://huggingface.co/datasets) and click **Create new dataset** (or [huggingface.co/new-dataset](https://huggingface.co/new-dataset)).
   - Choose a name, select your namespace, set visibility, then create.

2. **Add CSV files**
   - Evaluation expects CSV files with columns `t`, `y_t`, and `text` (see [Data format](#data-format)). Upload into the repo:
     - **Option A:** In the repo → **Files and versions** → **Add file** → **Upload files** (root or a subfolder like `data/`; then use `--datasets_hf_subdir data`).
     - **Option B:** CLI:
       ```bash
       huggingface-cli upload YOUR_USER/dataset-name ./path/to/csv_folder data --repo-type dataset
       ```
       Then run with `--datasets_hf YOUR_USER/dataset-name --datasets_hf_subdir data`.

3. **Optional:** Add a `README.md` in the dataset repo describing the CSVs.

4. **Use in this repo:** e.g. `--datasets_hf bekzatajan/fnspid --datasets_hf_subdir data` (for private repos, set `HF_TOKEN` or `huggingface-cli login`).

---

## Example notebooks

- [TTFM Inference Quick Start](notebooks/ttfm-inference-quickstart.ipynb) — Load the pipeline, fetch `data/aal_with_text.csv` from [bekzatajan/fnspid](https://huggingface.co/datasets/bekzatajan/fnspid), run `predict()`, and plot the forecast.

---

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

---

## Reference

### Data format

- **Input:** A directory of CSV files (or a Hugging Face dataset repo via `--datasets_hf`). Each CSV must have:
  - `t` — timestamp (e.g. `1980-01-31`)
  - `y_t` — target value (numeric)
  - `text` — per-step text context (can be empty string)

- **Discovery:** All `.csv` files under the given directory are used, except those whose basename ends with `_embeddings`, `_original`, `_temp`, `_results`, or `_temp_results`.

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
| `--eval_toto` | Toto (optional: `pip install synthefy-ttfm[toto]`) |

### Output layout

Under `--output_dir`:

```
<output_dir>/
  <datasets_dir_name>/
    context_<seq_len>/
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
bash scripts/post_eval.sh
# Or: bash scripts/post_eval.sh ./results/suite/context_64 scripts/latex/config_example.yaml
```

This writes `report/report.md` and bar plots. To only generate bar plots:

```bash
uv run python scripts/plot_bars.py --results_dir ./results/suite/context_64
```

See [Post-evaluation pipeline](#post-evaluation-pipeline) for scatter plots, qualitative forecasts, and LaTeX tables.

### TTFM and the LLM server

TTFM and `--eval_gpt_forecast` require an LLM server for context summarization or GPT forecasts. Start vLLM (or an OpenAI-compatible server) before running. Default URL: `http://localhost:8004/v1`, default model: `openai/gpt-oss-120b`. Override with `--llm_base_url` and `--llm_model`, or `VLLM_BASE_URL` and `VLLM_MODEL`. No API key is required for a local vLLM server.

**Quick start for TTFM eval:** In a separate terminal run `bash start_vllm.sh` (or `uv run vllm serve openai/gpt-oss-120b --port 8004`). Then run the evaluation CLI with `--eval_ttfmlf` and `--checkpoint`.

### Environment variables

| Variable | Meaning |
|----------|---------|
| `TTFM_CHECKPOINT` | Default HF repo id for TTFM checkpoint (used if `--checkpoint` is not set) |
| `TTFM_EVAL_DATASETS_DIR` | Default `--datasets_dir` |
| `TTFM_EVAL_OUTPUT_DIR` | Default `--output_dir` |
| `HF_TOKEN` | Hugging Face token for private model/dataset repos and TabPFN |
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

- **One-command report:** `bash scripts/post_eval.sh [results_dir] [latex_config]` — bar plots and `report/report.md`; optional LaTeX table.
- **Bar plots:** `scripts/plot_bars.py` — aggregate MAE, grouped bars, TTFM win rate, improvement %, ELO (if multielo installed).
- **Scatter plots:** `scripts/plot_scatter.py` — TTFM vs baseline MAE per sample.
- **Qualitative forecast plots:** `scripts/plot_qualitative_forecasts.py` — context + ground truth + forecasts for selected samples.
- **LaTeX tables:** `scripts/latex/generate_table.py` — stats to LaTeX.
- **LLM trend-direction eval:** `scripts/llm_trend_description_eval.py` — LLM-only direction accuracy on your CSVs.

See the script `--help` and the existing README sections in version control for full options.
