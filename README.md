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
