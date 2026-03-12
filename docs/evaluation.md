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

```bash
# FNSPID — prepared financial-news evaluation assets
uv run python scripts/download_data.py --dataset fnspid --csvs      # CSVs only
uv run python scripts/download_data.py --dataset fnspid --summaries # pre-computed summaries
uv run python scripts/download_data.py --dataset fnspid --all       # both

# Migas-1.5 ICML suite
uv run python scripts/download_data.py --dataset suite --csvs
uv run python scripts/download_data.py --dataset suite --all
```

- Default destinations: `data/fnspid_prepared/` (fnspid), `data/migas_1_5_suite/` (suite).
- Override with `--local_dir`. List presets: `--list`. Custom repo: `--repo_id org/my-dataset`.
- Hit 429 rate limits? Use `--max-workers 1` (default).

---

## Running evaluations

### Baselines only (no LLM)

```bash
uv run python -m migaseval.evaluation \
  --datasets_dir ./data/fnspid_prepared/fnspid_0.5_complement_csvs \
  --output_dir   ./results \
  --seq_len 384 --pred_len 16 --batch_size 64 \
  --eval_chronos2 --eval_timesfm --eval_prophet --eval_naive
```

### With Migas-1.5 (requires LLM server or pre-computed summaries)

```bash
uv run python -m migaseval.evaluation \
  --datasets_dir ./data/fnspid_prepared/fnspid_0.5_complement_csvs \
  --output_dir   ./results \
  --seq_len 384 --pred_len 16 --batch_size 64 \
  --eval_migas15 --eval_chronos2 --eval_timesfm
```

### Fast path: pre-computed summaries (no LLM server)

```bash
uv run python -m migaseval.evaluation \
  --datasets_dir  ./data/fnspid_prepared/fnspid_0.5_complement_csvs \
  --summaries_dir ./data/fnspid_prepared/fnspid_0.5_complement \
  --output_dir    ./results \
  --seq_len 384 --pred_len 16 --batch_size 64 \
  --eval_migas15 --checkpoint Synthefy/migas-1.5
```

### Cache-then-evaluate workflow

```bash
# Step 1 — cache summaries (requires LLM server)
uv run python -m migaseval.evaluation \
  --datasets_dir ./data/fnspid_prepared/fnspid_0.5_complement_csvs \
  --output_dir   ./results \
  --seq_len 384 --pred_len 16 --batch_size 64 \
  --cache_summaries

# Step 2 — evaluate from cache (no LLM)
uv run python scripts/eval_simple.py \
  --summaries_dir ./results/output/context_384 \
  --checkpoint    Synthefy/migas-1.5 \
  --context_lengths 32 64 128 256 384 \
  --eval_timesfm --eval_toto --eval_prophet
```

---

## Baselines

| Flag | Description |
|------|-------------|
| `--eval_migas15` | Migas-1.5 (default HF repo `Synthefy/migas-1.5`; override with `--checkpoint`) |
| `--eval_migas15_timesfm` | Migas-1.5 with TimesFM backbone (requires `--checkpoint_timesfm`) |
| `--eval_chronos2` | Chronos-2 univariate |
| `--eval_chronos2_multivar` | Chronos-2 with covariates |
| `--eval_chronos2_gpt` | Chronos-2 + GPT forecasts (run `--eval_gpt_forecast` first) |
| `--eval_gpt_forecast` | LLM-only forecast (needs vLLM / OpenAI-compatible server) |
| `--eval_timesfm` | TimesFM 2.5 univariate |
| `--eval_naive` | Naive (last value) |
| `--eval_prophet` | Prophet |
| `--eval_tabpfn` | TabPFN 2.5 (requires `HF_TOKEN`) |
| `--eval_toto` | Toto (optional: `pip install -e ".[toto]"`) |

---

## Output layout

```
<output_dir>/
  <datasets_dir_name>/
    context_<seq_len>/
      eval_meta.json
      stats_Context_<seq_len>_allsamples.csv
      stats_Context_<seq_len>_june2024plus.csv   # if applicable
      outputs/
        <dataset_name>/
          input.npy  gt.npy  migas_pred.npy  chronos_univar_pred.npy  ...
```

Re-runs skip datasets that already have all requested model outputs. Delete `outputs/<dataset_name>/` or use a new `--output_dir` to re-evaluate.

---

## Post-evaluation reports and plots

```bash
# Bar plots + report.md (single entry point)
uv run python scripts/post_eval.py --results_dir ./results/suite/context_64

# Add scatter plots
uv run python scripts/post_eval.py --results_dir ./results/suite/context_64 --scatter

# Add qualitative forecast plots
uv run python scripts/post_eval.py --results_dir ./results/suite/context_64 --qualitative

# Everything
uv run python scripts/post_eval.py --results_dir ./results/suite/context_64 --all
```

Individual scripts for more control: `scripts/plot_bars.py`, `scripts/plot_scatter.py`, `scripts/plot_qualitative_forecasts.py`. Use `--help` for options.

---

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `MIGAS_CHECKPOINT` | `Synthefy/migas-1.5` | Override the checkpoint source (HF repo id or local path) |
| `MIGAS_EVAL_DATASETS_DIR` | `./data/test` | Default `--datasets_dir` |
| `MIGAS_EVAL_OUTPUT_DIR` | — | Default `--output_dir` |
| `HF_TOKEN` | — | Hugging Face token (only needed for TabPFN) |
| `VLLM_BASE_URL` | `http://localhost:8004/v1` | LLM server URL for summarization |
| `VLLM_MODEL` | `openai/gpt-oss-120b` | Model name |
| `VLLM_TENSOR_PARALLEL_SIZE` | # GPUs in `CUDA_VISIBLE_DEVICES` | Tensor parallelism |

---

## Adding a new baseline

1. Add `src/migaseval/baselines/<name>.py` with:

   ```python
   def evaluate_<name>(loader, device, pred_len, **kwargs) -> dict:
       # returns {"input": Tensor, "gt": Tensor, "predictions": {model_name: Tensor}}
   ```

2. In `src/migaseval/baselines/__init__.py`:

   ```python
   register_baseline(name="<name>", eval_func=..., prediction_keys=[...], help_text="...")
   ```

The CLI gets `--eval_<name>` automatically. Baselines depending on another baseline's outputs are run after their dependencies.

---

## FNSPID dataset

[FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) (Dong et al., 2024) contains 10M+ financial news articles linked to stock symbols. Prepared assets are at [Synthefy/fnspid](https://huggingface.co/datasets/Synthefy/fnspid):

- `fnspid_0.5_complement_csvs/` — `t`, `y_t`, `text` CSVs ready for eval
- `fnspid_0.5_complement/` — cached LLM summaries (skip LLM server during eval)

<details>
<summary>Building FNSPID from raw data (advanced)</summary>

Use `scripts/prepare_fnspid.sh` and the scripts under `scripts/fnspid/`. Requires downloading the raw dataset from [Zihan1004/FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) and a running LLM server for the summarization step.

</details>
