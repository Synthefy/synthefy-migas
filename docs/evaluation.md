# Evaluation Guide

Full reference for running Migas-1.5 evaluations and generating reports.

---

## Data format

Each CSV or Parquet file must have:

| Column | Description |
|--------|-------------|
| `t` | Timestamp (e.g. `1980-01-31`) |
| `y_t` | Target value (numeric) |
| `text` | Per-step text context (can be empty string) |

Files ending in `_embeddings`, `_original`, `_temp`, `_results`, or `_temp_results` are skipped automatically.

---

## Downloading evaluation data

All evaluation data lives in a single Hugging Face repo: **Synthefy/multimodal_datasets**. Use `--dataset` to choose which folder to download:

| Choice | Contents | Default local path |
|--------|----------|--------------------|
| `fnspid` | FNSPID prepared financial-news assets | `data/fnspid_prepared/` |
| `suite` | Migas-1.5 ICML suite | `data/migas_1_5_suite/` |
| `subset` | Small set (2 time series) for quick runs | `data/subset/` |
| `all` | All of the above | `data/` |

```bash
# Subset — 2 datasets, good for notebooks and quick tests
uv run python -m migaseval.scripts.download_data --dataset subset --csvs
uv run python -m migaseval.scripts.download_data --dataset subset --all

# FNSPID — prepared financial-news evaluation assets
uv run python -m migaseval.scripts.download_data --dataset fnspid --csvs
uv run python -m migaseval.scripts.download_data --dataset fnspid --all

# Migas-1.5 ICML suite
uv run python -m migaseval.scripts.download_data --dataset suite --csvs
uv run python -m migaseval.scripts.download_data --dataset suite --all

# Everything
uv run python -m migaseval.scripts.download_data --dataset all --all
```

- Specify what to download: `--csvs`, `--summaries`, or `--all`. List presets: `--list`.
- Override destination: `--local_dir`.
- Hit 429 rate limits? Use `--max-workers 1` (default).

---

## Running evaluations

### From pre-computed summaries (no LLM server)

The fastest path uses pre-cached summary JSONs (downloaded from HF or generated earlier):

```bash
uv run python -m migaseval.evaluation \
  --summaries_dir ./data/subset/subset_migas15/subset \
  --output_dir    ./results \
```

### From raw CSVs (generates summaries, needs LLM server)

```bash
uv run python -m migaseval.evaluation \
  --datasets_dir ./data/subset/subset_migas15/subset_csvs \
  --output_dir   ./results \
```

### With additional baselines

```bash
uv run python -m migaseval.evaluation \
  --summaries_dir ./data/subset/subset_migas15/subset \
  --output_dir    ./results \
  --eval_timesfm --eval_prophet
```

### Context length sweeping

Evaluate at multiple context lengths in a single run:

```bash
uv run python -m migaseval.evaluation \
  --summaries_dir ./data/subset/subset_migas15/subset \
  --output_dir    ./results \
  --context_lengths 32 64 128 256 384 \
  --eval_timesfm --eval_toto --eval_prophet
```

---

## Baselines

Migas-1.5 (with Chronos backbone) always runs. Additional baselines are enabled with flags:

| Flag | Description |
|------|-------------|
| `--eval_timesfm` | TimesFM 2.5 univariate |
| `--eval_prophet` | Prophet |
| `--eval_tabpfn` | TabPFN 2.5 (requires `HF_TOKEN`) |
| `--eval_toto` | Toto (optional: `pip install -e ".[toto]"`) |
| `--eval_sarima` | Seasonal ARIMA (auto_arima) |

---

## Output layout

```
<output_dir>/
  context_<ctx_len>/
    results_<test_set>_ctx<ctx_len>.csv
    predictions/
      <dataset_name>/
        migas15.npz  chronos.npz  timesfm.npz  prophet.npz  ...
```

Each `.npz` file contains `history`, `predictions`, `gt`, `history_means`, `history_stds`. Re-runs automatically load cached predictions per model — only missing models are computed.

---

## Post-evaluation reports and plots

```bash
# Bar plots + report.md (single entry point)
uv run python -m migaseval.scripts.post_eval --results_dir ./results/suite/context_64

# Add scatter plots
uv run python -m migaseval.scripts.post_eval --results_dir ./results/suite/context_64 --scatter

# Add qualitative forecast plots
uv run python -m migaseval.scripts.post_eval --results_dir ./results/suite/context_64 --qualitative

# Everything
uv run python -m migaseval.scripts.post_eval --results_dir ./results/suite/context_64 --all
```

Individual script modules for more control: `python -m migaseval.scripts.plot_bars`, `python -m migaseval.scripts.plot_scatter`, `python -m migaseval.scripts.plot_qualitative_forecasts`. Use `--help` for options.

---

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `MIGAS_EVAL_DATASETS_DIR` | `./data/test` | Default `--datasets_dir` |
| `MIGAS_EVAL_OUTPUT_DIR` | — | Default `--output_dir` |
| `HF_TOKEN` | — | Hugging Face token (only needed for TabPFN) |
| `VLLM_BASE_URL` | `http://localhost:8004/v1` | LLM server URL for summarization |
| `VLLM_MODEL` | `openai/gpt-oss-120b` | Model name |
| `VLLM_TENSOR_PARALLEL_SIZE` | # GPUs in `CUDA_VISIBLE_DEVICES` | Tensor parallelism |

---

## Adding a new baseline

1. Add an `evaluate_<name>_precomputed()` function in `src/migaseval/eval_utils.py`.
2. Add the `--eval_<name>` flag and call in `src/migaseval/evaluation.py`.

---

## FNSPID dataset

[FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) (Dong et al., 2024) contains 10M+ financial news articles linked to stock symbols. Prepared assets are in [Synthefy/multimodal_datasets](https://huggingface.co/datasets/Synthefy/multimodal_datasets) under the `fnspid_migas15/` folder:

- `fnspid_migas15/fnspid_0.5_complement_csvs/` — `t`, `y_t`, `text` CSVs ready for eval
- `fnspid_migas15/fnspid_0.5_complement/` — cached LLM summaries (skip LLM server during eval)

Download with: `uv run python -m migaseval.scripts.download_data --dataset fnspid --all`. The **subset** option (2 time series) is smaller and suited to notebooks: `--dataset subset --all`.

<details>
<summary>Building FNSPID from raw data (advanced)</summary>

Use `scripts/prepare_fnspid.sh` and the scripts under `scripts/fnspid/`. Requires downloading the raw dataset from [Zihan1004/FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) and a running LLM server for the summarization step.

</details>
