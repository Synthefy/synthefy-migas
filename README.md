# TTFM Eval

Evaluation infrastructure for the **TTFM** (Text-and-Time-Series Fusion Model) and forecasting baselines. You only need a checkpoint and CSV datasets.

## Quick start (univariate baselines only)

To run **without** a TTFM checkpoint (Chronos-2, TimesFM, etc. — no LLM required):

1. **Install** (from repo root):

   ```bash
   uv sync
   # or: pip install -e .
   ```

2. **Run evaluation** (example: Chronos-2 + TimesFM):

   ```bash
   uv run python -m ttfmeval.evaluation \
     --datasets_dir ./data/test \
     --output_dir ./results \
     --seq_len 64 \
     --pred_len 16 \
     --batch_size 64 \
     --eval_chronos2 \
     --eval_timesfm
   ```

   Or use the wrapper script (same baselines by default):

   ```bash
   bash run.sh
   ```

   `run.sh` does **not** enable TTFM by default. To also evaluate TTFM, see [Evaluating your TTFM model](#evaluating-your-ttfm-model) below.

## Evaluating a TTFM model

To evaluate **a TTFM checkpoint** (text-conditioned forecasts), you need three things:

1. **Checkpoint** — Your saved TTFM model (`.pt` file) compatible with this package’s model architecture (e.g. `state_dict` from the same layout).
2. **Data** — CSVs with columns `t`, `y_t`, `text` (see [Data format](#data-format)).
3. **LLM server** — TTFM uses an LLM for context summarization. You must start a vLLM (or OpenAI-compatible) server **before** running eval. No API key is needed when using a local server.

### 1. Start the LLM server

In a **separate terminal**, start the server so it is running when you launch evaluation. If you use the script, install the optional `vllm` extra first:

```bash
uv sync --extra vllm
```

**Option A — Use the provided script** (set `VLLM_MODEL`, `VLLM_PORT`, `CUDA_VISIBLE_DEVICES` in the environment or edit `start_vllm.sh` if needed). For large models (e.g. 120B), set `CUDA_VISIBLE_DEVICES` to the GPUs you want (e.g. `0,1,2,3`); the script will use tensor parallelism across them so the model fits.

```bash
bash start_vllm.sh
```

**Option B — Run vLLM manually** (from repo root after `uv sync --extra vllm`):

```bash
uv run vllm serve openai/gpt-oss-120b --port 8004
```

Use a smaller model if you prefer (e.g. another Hugging Face model); set `--llm_model` / `VLLM_MODEL` when running eval to match.

By default the eval script expects the server at `http://localhost:8004/v1` and model `openai/gpt-oss-120b`. Override with `--llm_base_url` and `--llm_model`, or `VLLM_BASE_URL` and `VLLM_MODEL`.

### 2. Run evaluation with TTFM

Set your checkpoint path (env or CLI) and enable TTFM:

```bash
export TTFM_CHECKPOINT=/path/to/your/ttfm_checkpoint.pt

uv run python -m ttfmeval.evaluation \
  --datasets_dir ./data/test \
  --output_dir ./results \
  --seq_len 64 \
  --pred_len 16 \
  --batch_size 64 \
  --eval_ttfmlf \
  --eval_chronos2 \
  --eval_timesfm \
  --checkpoint "$TTFM_CHECKPOINT"
```

Or use the wrapper script and pass the checkpoint:

```bash
bash run.sh --eval_ttfmlf --checkpoint /path/to/your/ttfm_checkpoint.pt
```

If the server is not running or the URL/model is wrong, TTFM evaluation will fail when it tries to call the LLM.

## Data format

- **Input:** A directory of CSV files (e.g. `./data/test`). Each CSV must have:
  - `t` — timestamp (e.g. `1980-01-31`)
  - `y_t` — target value (numeric)
  - `text` — per-step text context (can be empty string)

- **Discovery:** All `.csv` files under the given directory are used, except those whose basename ends with `_embeddings`, `_original`, `_temp`, `_results`, or `_temp_results`.

## Baselines

Enable with `--eval_<name>`:

| Flag | Description |
|------|-------------|
| `--eval_ttfmlf` | TTFM model (requires `--checkpoint` or `TTFM_CHECKPOINT`) |
| `--eval_ttfmlf_timesfm` | TTFM with TimesFM as univariate (requires `--checkpoint_timesfm`) |
| `--eval_chronos2` | Chronos-2 univariate |
| `--eval_chronos2_multivar` | Chronos-2 with covariates |
| `--eval_chronos2_gpt` | Chronos-2 + GPT forecasts (run `--eval_gpt_forecast` first or have cached) |
| `--eval_gpt_forecast` | LLM-only forecast (needs vLLM / OpenAI-compatible server) |
| `--eval_timesfm` | TimesFM 2.5 univariate |
| `--eval_naive` | Naive (last value) |
| `--eval_prophet` | Prophet |
| `--eval_tabpfn` | TabPFN 2.5 time-series (requires `HF_TOKEN`) |
| `--eval_toto` | Toto (optional: `pip install ttfm-eval[toto]`) |

## Output layout

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

- **Caching:** Re-runs skip datasets that already have all requested model outputs; use a new `--output_dir` or delete `outputs/<dataset_name>/` to re-evaluate.

### After evaluation: report and plots

From the repo root you can generate a **Markdown report** and **bar plots** in one step:

```bash
# Default: results/suite/context_64
bash scripts/post_eval.sh

# Or specify results dir and optional LaTeX config
bash scripts/post_eval.sh ./results/suite/context_64 scripts/latex/config_example.yaml
```

This writes `results/<suite>/context_<seq_len>/report/report.md` (with links to bar plots and optional LaTeX table). To only generate bar plots:

```bash
uv run python scripts/plot_bars.py --results_dir ./results/suite/context_64
```

Bar plots include: aggregate MAE by model, grouped bars per dataset, TTFM win rate per dataset, improvement-over-baseline, and optional ELO ratings (if `multielo` is installed). See [Post-evaluation pipeline](#post-evaluation-pipeline) for scatter plots, qualitative forecasts, and LaTeX tables.

## TTFM and the LLM server

**TTFM** and **`--eval_gpt_forecast`** require an LLM server for context summarization or GPT forecasts. You must start it yourself (see [Evaluating your TTFM model](#evaluating-your-ttfm-model)). Default URL: `http://localhost:8004/v1`, default model: `openai/gpt-oss-120b`. Override with `--llm_base_url` and `--llm_model`, or `VLLM_BASE_URL` / `VLLM_MODEL`. No API key is required for a local vLLM server.

## Environment variables

| Variable | Meaning |
|----------|---------|
| `TTFM_CHECKPOINT` | Default path to TTFM checkpoint (used if `--checkpoint` is not set) |
| `TTFM_EVAL_DATASETS_DIR` | Default `--datasets_dir` |
| `TTFM_EVAL_OUTPUT_DIR` | Default `--output_dir` |
| `VLLM_BASE_URL` | vLLM/LLM server URL for TTFM summarizer (default `http://localhost:8004/v1`) |
| `VLLM_MODEL` | Model name for TTFM summarizer |
| `VLLM_TENSOR_PARALLEL_SIZE` | GPUs for tensor parallelism (default: number of GPUs in `CUDA_VISIBLE_DEVICES`) |
| `HF_TOKEN` | Hugging Face token for TabPFN (gated model) |

## Adding a new baseline

1. Add a module under `src/ttfmeval/baselines/` with a function:

   `evaluate_<name>(loader, device, pred_len, **kwargs) -> dict`

   returning `{"input": Tensor, "gt": Tensor, "predictions": {model_name: Tensor}}`.

2. In `src/ttfmeval/baselines/__init__.py`, import it and call:

   `register_baseline(name="<name>", eval_func=..., prediction_keys=[...], help_text="...")`.

The CLI will automatically get a `--eval_<name>` flag.

### Baseline evaluation order (dependencies)

Baselines that **depend on another baseline’s outputs** are run after their dependencies. For example, `chronos2_gpt` needs precomputed `gpt_forecast` predictions. The evaluator sorts the requested baselines so that any baseline whose `depends_on` is also in the requested set runs **after** that dependency. So you get a deterministic order: no-dep baselines first (e.g. `gpt_forecast`, `chronos2`, `timesfm`), then dependents (e.g. `chronos2_gpt`). That way cached or just-computed `gpt_forecast_pred.npy` is available when `chronos2_gpt` runs.

---

## Post-evaluation pipeline

Scripts under `scripts/` expect the same output layout as main eval (e.g. `results/<suite>/context_64/` with `outputs/<dataset>/` and `stats_Context_64_allsamples.csv`). Scripts that expect `per_dataset_metrics.csv` fall back to `stats_Context_*_allsamples.csv`; the `n_eval_samples` column is used when present.

After running the main evaluation, you can use:

- **One-command report**: `bash scripts/post_eval.sh [results_dir] [latex_config]` — runs bar plots and writes `report/report.md`; optionally runs LaTeX table generation.
- **Bar plots**: `scripts/plot_bars.py` — aggregate MAE by model, grouped bars per dataset, TTFM win rate, improvement %, ELO (if multielo installed). Outputs go to `results_dir/report/` by default.
- **Scatter plots**, **qualitative forecast plots**, **LaTeX tables**, and **LLM trend-direction** (see below).

### 1. Sample-level scatter plots (TTFM vs another model)

Compares TTFM MAE vs a chosen baseline’s MAE at sample level. Produces per-dataset scatter PDFs/PNGs, a multi-panel summary figure, an overall scatter plot, and a correlations CSV.

```bash
# From repo root (ensure PYTHONPATH includes . or run with uv)
uv run python scripts/plot_scatter.py \
  --results_dir ./results/my_suite/context_64 \
  --compare_model chronos_univar
```

- **`--results_dir`**: Directory that contains `outputs/` and a stats CSV. The script looks for `per_dataset_metrics.csv` first, then falls back to `stats_Context_*_allsamples.csv`.
- **`--compare_model`**: Baseline to compare (e.g. `chronos_univar`, `timesfm_univar`, `gpt_forecast`).
- **`--window_length`**: Optional; restrict to a sliding window where TTFM has the largest advantage over the `timeseries` baseline (requires `timeseries_pred.npy` in outputs).

Requires `matplotlib`. Outputs go under `results_dir` (e.g. `sample_scatter_plots_ttfm_vs_chronos_univar/`, `sample_scatter_overall_ttfm_vs_chronos_univar.pdf`, and a correlations CSV).

### 2. Qualitative forecast plots

Finds samples where a chosen model beats a baseline (or absolute best, or worst) and plots context + ground truth + model forecasts. Supports dates on the x-axis, grid plots, and optional LLM-generated context summaries (requires `src.context_summary.ContextSummarizer`; if missing, `--generate_summaries` is skipped). Uses `ttfmeval.dataset` for normalization; dataset CSVs are resolved from `--datasets_dir` (including subdirs).

```bash
uv run python scripts/plot_qualitative_forecasts.py \
  --results_dir ./results/my_suite/context_64 \
  --datasets_dir ./data/test \
  --output_dir ./results/my_suite/context_64/qualitative_plots \
  --context_len 64 --pred_len 16 \
  --baseline_model chronos_univar \
  --top_k 5
```

- **`--results_dir`**: Eval output dir containing `outputs/` and a stats CSV (uses `stats_Context_*_allsamples.csv` if `per_dataset_metrics.csv` is missing).
- **`--datasets_dir`**: Directory containing the dataset CSVs (used for normalization and optional dates); subdirs are searched for `<dataset_name>.csv`.
- **`--output_dir`**: Where to save PDF/PNG plots (default: `results_dir/qualitative_plots`).
- **`--better_model` / `--worse_model`**: Which model should win (default: ttfm vs chronos_univar). Use `--absolute_best` to pick samples with lowest MAE for one model, or `--show_worst` to show where the “better” model loses.
- **`--models_to_plot`**: Comma-separated list of models to draw (e.g. `ttfm,chronos_univar`).
- **`--create_grid`** / **`--grid_only`**: Create multi-sample grid figures per dataset.
- **`--use_dates`**: Use real dates from CSVs on the x-axis.
- **`--generate_summaries`**: Add LLM context summaries below plots (optional; uses `ttfmeval.model.util.ContextSummarizer` when available, else `src.context_summary`).

Requires `matplotlib`, `torch`. Saves individual PDF/PNG per sample and optionally `grid.pdf` and `sample_metadata.csv`.

### 3. LaTeX results table and plots

Turns eval stats CSVs into LaTeX tables (main table, context-mean table, optional dataset summary tables) and optional improvement/ELO plots. Use a YAML config with `context_groups` (or `horizon_groups`) and `csv_path` pointing at your `stats_Context_*_allsamples.csv`. Paths in the config are resolved relative to the config file.

```bash
cd scripts/latex
# config_example.yaml uses context_groups and models: auto (infer from CSV)
uv run generate_table.py --config config_example.yaml --output results_table.tex --standalone
```

- **Config**: In `context_groups` set `csv_path` (relative or absolute) and `context_length`. Use `models: auto` to infer models from CSV columns, or list them explicitly. Optional: `dataset_display_names`, `datasets`, `precision`, `caption`, `label`. See `config_example.yaml` and `config_merged_cur.yaml` for examples.
- **`--standalone`**: Writes standalone `.tex` and compiles to PDF when `pdflatex` is available.
- **`--plot`**: Generate combined ELO and improvement bar charts (saved under `--plot_dir`).
- **`--summary`** / **`--summary-grouped`**: Dataset summary LaTeX tables (optional `--data_dir`).

To build the PDF from the shell script:

```bash
CONFIG=my_config.yaml bash generate_pdf.sh
```

Requires `pyyaml`, `pandas`. Optional: `pdflatex` for PDF output.

### 4. LLM trend-direction evaluation (optional)

Separate pipeline that evaluates an **LLM only** on trend direction (up/down/flat) over your CSV datasets: one-step direction and mean-horizon direction. Does **not** use TTFM or the main eval outputs; it only needs CSVs with `t`, `y_t`, `text` and an OpenAI-compatible LLM server.

```bash
# Start vLLM (or another OpenAI-compatible server) first, then:
uv run python scripts/llm_trend_description_eval.py \
  --csv_dir ./data/test \
  --context_length 64 --horizon 16 \
  --include_text \
  --output_dir ./trend_desc_eval/results
```

- **`--llm_base_url`** / **`--llm_model`**: Default `http://localhost:8004/v1` and `openai/gpt-oss-120b` to match this repo’s vLLM default.
- **`--include_text`**: Include the `text` column in the context sent to the LLM.

Requires `openai`, `pandas`, `numpy`, `tqdm`. Writes `unified_eval_results.json` and `direction_details.csv` under `--output_dir`.
