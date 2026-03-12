# %% [markdown]
# # Bring Your Own Data
# 
# This notebook shows how to run Migas-1.5 inference on **your own data**. You'll learn how to prepare a DataFrame with the required columns (`t`, `y_t`, `text`), use `predict_from_dataframe()` for quick single-sample forecasting, and construct raw inputs for full control.
# 
# **Requirements:** Install the package from the repo root (`uv sync`). For live summarization a vLLM server must be running (see README); alternatively pass pre-computed `summaries` to skip LLM calls.
# 
# **See also:** [Migas-1.5 Inference Quick Start](migas-1.5-inference-quickstart.ipynb) for a minimal example with prepared FNSPID data.

# %%
%matplotlib inline
import warnings
warnings.filterwarnings("ignore", message="IProgress not found")

import numpy as np
import pandas as pd
import torch
from migaseval import MigasPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = MigasPipeline.from_pretrained("Synthefy/migas-1.5", device=device)
print(f"Using device: {device}")

# %% [markdown]
# ## Data format
# 
# Migas-1.5 expects a DataFrame (or CSV/Parquet file) with three columns:
# 
# | Column | Type | Description |
# |--------|------|-------------|
# | `t` | str / datetime | Timestamp for each row |
# | `y_t` | float | Target value (the time series) |
# | `text` | str | Per-timestep text context (news, events, etc.). Use `""` when no text is available. |
# 
# Below we create a small synthetic DataFrame to demonstrate the workflow. Replace this with your own data.

# %%
np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=100, freq="D")
values = np.cumsum(np.random.randn(100) * 0.5) + 100

df = pd.DataFrame({
    "t": dates.strftime("%Y-%m-%d"),
    "y_t": values,
    "text": [""] * 100,
})

# Attach text context to a few dates
df.loc[40, "text"] = "Company reports strong Q1 earnings, beating estimates by 12%."
df.loc[70, "text"] = "FDA approves new drug; analysts raise price target."

df.head()

# %% [markdown]
# ## Quick forecast with `predict_from_dataframe`
# 
# The simplest path: pass a DataFrame and get a 1-D numpy array back.

# %%
import matplotlib.pyplot as plt

pred_len = 16
forecast = pipeline.predict_from_dataframe(df, pred_len=pred_len)

print(f"Forecast shape: {forecast.shape}")  # (16,)

plt.figure(figsize=(10, 3))
plt.plot(df["y_t"].values, label="context")
plt.plot(np.arange(len(df), len(df) + pred_len), forecast, label="forecast")
plt.legend()
plt.xlabel("Time step")
plt.title("Migas-1.5 forecast on custom data")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Using a subset of context
# 
# If your DataFrame is long, use `seq_len` to keep only the last N rows as context. Shorter context is faster; longer context gives the model more history.

# %%
forecast_short = pipeline.predict_from_dataframe(df, pred_len=16, seq_len=64)
print(f"Forecast (64-step context): {forecast_short.shape}")

# %% [markdown]
# ## Manual input construction
# 
# For full control (e.g. batching multiple samples), construct the raw `predict()` inputs yourself. The required shapes are:
# 
# - `context`: numpy array or torch tensor of shape `(B, T)`
# - `text`: `List[List[str]]` — one list of T strings per sample in the batch
# - `pred_len`: forecast horizon (int)

# %%
context = df["y_t"].values.astype(np.float32).reshape(1, -1)  # (1, 100)
text = [df["text"].fillna("").astype(str).tolist()]            # 1 x 100 strings

forecast_raw = pipeline.predict(context, text, pred_len=16)
print(f"Raw predict shape: {forecast_raw.shape}")  # torch.Size([1, 16, 1])

# %% [markdown]
# ## Loading from a CSV or Parquet file
# 
# If your data is already in a file with `t`, `y_t`, `text` columns, load it with `read_datafile` and pass the DataFrame directly.

# %%
from migaseval import read_datafile

# Example: load a prepared FNSPID CSV (download first: uv run python scripts/download_data.py --dataset fnspid --csvs)
csv_path = "../data/fnspid_prepared/fnspid_0.5_complement_csvs/adbe_with_text.csv"
df_csv = read_datafile(csv_path)

forecast_csv = pipeline.predict_from_dataframe(df_csv, pred_len=16, seq_len=384)
print(f"Forecast from CSV: {forecast_csv.shape}")

# %%



