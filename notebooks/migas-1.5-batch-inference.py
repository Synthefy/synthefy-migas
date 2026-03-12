# %% [markdown]
# # Batch Inference
# 
# This notebook shows how to run Migas-1.5 on **multiple time series** in a single `predict()` call and how to iterate over a directory of CSV files to collect forecasts.
# 
# **Requirements:** Install the package (`uv sync`). For live summarization a vLLM server must be running; to skip it, pass `summaries=` (see the [Offline Summaries](migas-1.5-offline-summaries.ipynb) notebook).
# 
# **Data:** Download prepared FNSPID CSVs and summaries:
# ```bash
# uv run python scripts/download_data.py --dataset fnspid --all
# ```

# %%
%matplotlib inline
import warnings
warnings.filterwarnings("ignore", message="IProgress not found")

import json, os
import numpy as np
import pandas as pd
import torch
from migaseval import MigasPipeline, list_data_files

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = MigasPipeline.from_pretrained("Synthefy/migas-1.5", device=device)
print(f"Using device: {device}")

# %% [markdown]
# ## Batching multiple samples in one call
# 
# `predict()` accepts a batch dimension: `context` has shape `(B, T)` and `text` is a list of B lists. This is more efficient than calling `predict()` in a loop.

# %%
csv_dir = "../data/fnspid_prepared/fnspid_0.5_complement_csvs"
csv_files = list_data_files(csv_dir)[:4]  # first 4 datasets
print(f"Using {len(csv_files)} datasets: {[os.path.basename(f) for f in csv_files]}")

seq_len = 384
pred_len = 16

contexts = []
texts = []
names = []

for path in csv_files:
    df = pd.read_csv(path)
    df = df.head(seq_len)
    contexts.append(df["y_t"].values.astype(np.float32))
    texts.append(df["text"].fillna("").astype(str).tolist())
    names.append(os.path.splitext(os.path.basename(path))[0])

context_batch = np.stack(contexts)  # (B, seq_len)
print(f"Batch context shape: {context_batch.shape}")

# %%
forecast_batch = pipeline.predict(context_batch, texts, pred_len=pred_len)
print(f"Batch forecast shape: {forecast_batch.shape}")  # (B, pred_len, 1)

# %% [markdown]
# ## Plotting batch results

# %%
import sys
sys.path.insert(0, "..")
from scripts.plotting_utils import plot_forecast_grid
import matplotlib.pyplot as plt

context_len = context_batch.shape[1]
history_2d = context_batch
preds_2d = {"Migas-1.5": forecast_batch[:, :, 0].detach().cpu().numpy()}
fig, axes = plot_forecast_grid(history_2d, None, preds_2d, context_len, pred_len, list(range(len(names))), titles=names)
plt.show()

# %% [markdown]
# ## Iterating over a directory with pre-computed summaries
# 
# For large-scale runs you'll typically loop over files and collect results. Using pre-computed summaries avoids the LLM bottleneck.

# %%
csv_dir = "../data/fnspid_prepared/fnspid_0.5_complement_csvs"
summaries_root = "../data/fnspid_prepared/fnspid_0.5_complement"

results = []

for csv_path in list_data_files(csv_dir)[:8]:
    name = os.path.splitext(os.path.basename(csv_path))[0]
    summary_path = os.path.join(summaries_root, name, "summary_0.json")

    if not os.path.exists(summary_path):
        print(f"  Skipping {name} (no summary found)")
        continue

    df = pd.read_csv(csv_path)
    with open(summary_path) as f:
        summary = json.load(f)["summary"]

    forecast = pipeline.predict_from_dataframe(
        df, pred_len=pred_len, seq_len=seq_len, summaries=[summary]
    )
    results.append({"dataset": name, "forecast": forecast.tolist()})
    print(f"  {name}: forecast={forecast[:3]}...")

print(f"\nCollected forecasts for {len(results)} datasets.")

# %% [markdown]
# ## Collecting results into a DataFrame

# %%
rows = []
for r in results:
    for step, val in enumerate(r["forecast"]):
        rows.append({"dataset": r["dataset"], "step": step, "forecast": val})

results_df = pd.DataFrame(rows)
print(results_df.head(20))

# Optionally save
# results_df.to_csv("forecasts.csv", index=False)

# %%



