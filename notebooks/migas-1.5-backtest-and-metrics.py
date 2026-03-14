# %% [markdown]
# # Backtest and Metrics
# 
# This notebook runs a **rolling-window backtest** on a single dataset: slide a context window through the time series, forecast the next `pred_len` steps with both **Migas-1.5** and **Chronos-2**, compare against ground truth, and compute standard metrics (MAE, MSE, MAPE, directional accuracy).
# 
# **Requirements:** Install the package (`uv sync`). For live summarization a vLLM server must be running; to skip it, use pre-computed summaries (see the offline section at the end of this notebook). See [LLM server setup](../README.md#optional-llm-server) in the README.
# 
# **Data:** Download the **subset**:
# ```bash
# uv run python -m migaseval.scripts.download_data --dataset subset --csvs
# ```
# for running the notebook using pre-computed summaries, you need to download the summaries:
# ```bash
# uv run python -m migaseval.scripts.download_data --dataset subset --summaries
# ```

# %% [markdown]
# > **Reproducing our benchmark results?** See the [Evaluation Guide](../docs/evaluation.md) for
# > the full CLI reference, baseline flags, output layout, and post-eval report scripts.

# %%
%matplotlib inline
import warnings
warnings.filterwarnings("ignore", message="IProgress not found")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json, os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from migaseval import MigasPipeline
from migaseval.model.inference_utils import evaluate_chronos
from migaseval.plotting_utils import apply_migas_style, plot_forecast_grid

apply_migas_style()

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = MigasPipeline.from_pretrained("Synthefy/migas-1.5", device=device)
print(f"Using device: {device}")

# %% [markdown]
# ## Configure the backtest

# %%
csv_path = "../data/subset/subset_migas15/subset_csvs/jpm_with_text.csv"  # one of 2 subset datasets
df = pd.read_csv(csv_path)
print(f"Dataset: {os.path.basename(csv_path)}  —  {len(df)} rows")

seq_len = 64
pred_len = 16
stride = pred_len  # non-overlapping windows; set to 1 for fully rolling

n_windows = (len(df) - seq_len - pred_len) // stride + 1
print(f"seq_len={seq_len}, pred_len={pred_len}, stride={stride}  →  {n_windows} windows")

# %% [markdown]
# ## Run rolling backtest

# %%
migas_forecasts = []
chronos_forecasts = []
ground_truths = []
last_context_values = []

for i in range(n_windows):
    start = i * stride
    end = start + seq_len

    df_ctx = df.iloc[start:end]
    gt = df["y_t"].values[end : end + pred_len]

    if len(gt) < pred_len:
        break

    # Migas-1.5 forecast
    migas_pred = pipeline.predict_from_dataframe(df_ctx, pred_len=pred_len)

    # Chronos-2 forecast (reuses the model already loaded by the pipeline)
    ctx_tensor = torch.tensor(
        df_ctx["y_t"].values.astype(np.float32)
    ).reshape(1, -1, 1)
    chronos_pred = evaluate_chronos(ctx_tensor, pred_len=pred_len, device=device)
    chronos_pred = chronos_pred[0, :, 0].detach().cpu().numpy()

    migas_forecasts.append(migas_pred)
    chronos_forecasts.append(chronos_pred)
    ground_truths.append(gt.astype(np.float32))
    last_context_values.append(float(df_ctx["y_t"].values[-1]))

    if (i + 1) % 10 == 0 or i == n_windows - 1:
        print(f"  Window {i + 1}/{n_windows}")

migas_forecasts = np.stack(migas_forecasts)        # (N, pred_len)
chronos_forecasts = np.stack(chronos_forecasts)  # (N, pred_len)
ground_truths = np.stack(ground_truths)          # (N, pred_len)
last_ctx = np.array(last_context_values)         # (N,)

print(f"\nBacktest complete: {len(migas_forecasts)} windows")

# %% [markdown]
# ## Compute metrics

# %%
def compute_backtest_metrics(preds, ground_truths, last_ctx):
    """Compute MAE, MSE, MAPE, and directional accuracy."""
    errors = preds - ground_truths
    mae_per_sample = np.mean(np.abs(errors), axis=1)
    mse_per_sample = np.mean(errors ** 2, axis=1)
    mape_per_sample = np.mean(
        np.abs(errors) / (np.abs(ground_truths) + 1e-8), axis=1
    ) * 100
    pred_dir = np.sign(preds[:, 0] - last_ctx)
    gt_dir = np.sign(ground_truths[:, 0] - last_ctx)
    valid = gt_dir != 0
    dir_acc = (pred_dir[valid] == gt_dir[valid]).mean() * 100 if valid.any() else float("nan")
    return {
        "MAE (mean)": np.mean(mae_per_sample),
        "MAE (median)": np.median(mae_per_sample),
        "MSE (mean)": np.mean(mse_per_sample),
        "MSE (median)": np.median(mse_per_sample),
        "MAPE % (mean)": np.mean(mape_per_sample),
        "MAPE % (median)": np.median(mape_per_sample),
        "Dir. acc % (step 1)": dir_acc,
    }

migas_metrics = compute_backtest_metrics(migas_forecasts, ground_truths, last_ctx)
chronos_metrics = compute_backtest_metrics(chronos_forecasts, ground_truths, last_ctx)

metrics_df = pd.DataFrame({"Migas-1.5": migas_metrics, "Chronos-2": chronos_metrics})
metrics_df.index.name = "metric"
print(metrics_df.to_string())

migas_mae_per_window = np.mean(np.abs(migas_forecasts - ground_truths), axis=1)
show_indices = np.argsort(migas_mae_per_window)[:4]
print(f"\nTop 4 lowest-MAE window indices: {show_indices.tolist()}")

# %% [markdown]
# ## Forecast vs ground truth — sample windows
# 
# Plot a few individual windows so you can visually inspect forecast quality.

# %%
tail = 60  # how many context steps to show before the forecast
history_2d = np.array([df["y_t"].values[idx * stride + seq_len - tail : idx * stride + seq_len] for idx in show_indices])
gt_2d = ground_truths[show_indices]
preds_2d = {"Migas-1.5": migas_forecasts[show_indices], "Chronos-2": chronos_forecasts[show_indices]}
fig, axes = plot_forecast_grid(history_2d, gt_2d, preds_2d, tail, pred_len, list(range(len(show_indices))), titles=None)
plt.show()

# %% [markdown]
# ## Offline backtest with pre-computed summaries
# 
# Pre-computed summaries from the subset use **stride=1** (one summary per rolling position). To match indices correctly, set `stride=1` above and pass the corresponding summary for each window. Below is a quick example loading the first few summaries.

# %%
dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
summaries_dir = f"../data/subset/subset_migas15/subset/{dataset_name}"

offline_migas = []
offline_chronos = []
offline_gts = []
offline_last_ctx = []
n_offline = min(10, n_windows)  # demo with first 10 windows

for i in range(n_offline):
    start = i * stride
    end = start + seq_len
    gt = df["y_t"].values[end : end + pred_len].astype(np.float32)
    if len(gt) < pred_len:
        break

    summary_idx = start
    summary_path = os.path.join(summaries_dir, f"summary_{summary_idx}.json")
    if not os.path.exists(summary_path):
        print(f"  Window {i}: summary not found at {summary_path}, skipping")
        continue

    with open(summary_path) as f:
        summary = json.load(f)["summary"]

    df_ctx = df.iloc[start:end]
    context = df_ctx["y_t"].values.astype(np.float32).reshape(1, -1)

    migas_pred = pipeline.predict(context, pred_len=pred_len, summaries=[summary])
    ctx_tensor = torch.tensor(context).reshape(1, -1, 1)
    chronos_pred = evaluate_chronos(ctx_tensor, pred_len=pred_len, device=device)

    offline_migas.append(migas_pred[0, :, 0].detach().cpu().numpy())
    offline_chronos.append(chronos_pred[0, :, 0].detach().cpu().numpy())
    offline_gts.append(gt)
    offline_last_ctx.append(float(context[0, -1]))

if offline_migas:
    offline_migas = np.stack(offline_migas)
    offline_chronos = np.stack(offline_chronos)
    offline_gts = np.stack(offline_gts)
    offline_last_ctx = np.array(offline_last_ctx)
    m_migas15 = compute_backtest_metrics(offline_migas, offline_gts, offline_last_ctx)
    m_chron = compute_backtest_metrics(offline_chronos, offline_gts, offline_last_ctx)
    print(pd.DataFrame({"Migas-1.5 (offline)": m_migas15, "Chronos-2": m_chron}).to_string())
else:
    print("No summaries found — download them first: uv run python -m migaseval.scripts.download_data --dataset subset --summaries")

# %%



