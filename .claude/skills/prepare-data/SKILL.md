---
name: prepare-data
description: "Use this skill any time data is loaded, fetched, downloaded, or structured for the Migas-1.5 pipeline. This includes: loading CSVs, fetching live market data via yfinance, formatting DataFrames with t/y_t/text columns, splitting context and ground-truth windows, or preparing batch inference inputs. Trigger whenever the user mentions data prep, data loading, downloading tickers, or formatting series data."
---

# Prepare Data for Migas-1.5

## Required CSV Format

Migas-1.5 expects a CSV (or DataFrame) with these columns:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `t` | string | Yes | Date in `YYYY-MM-DD` format |
| `y_t` | float | Yes | Numeric value (price, metric, etc.) |
| `text` | string | No | Per-timestep text (news headlines, etc.). Use `""` if none. |

The `text` column is optional — if absent, the pipeline fills it with empty strings.

## Loading and Splitting

The standard pattern is to take the last `SEQ_LEN + PRED_LEN` rows, using the
first `SEQ_LEN` as context and the remaining `PRED_LEN` as ground truth:

```python
import pandas as pd
import numpy as np

SEQ_LEN  = 64   # context window (steps of history)
PRED_LEN = 16   # forecast horizon

raw = pd.read_csv("your_data.csv")
raw["t"] = pd.to_datetime(raw["t"]).dt.strftime("%Y-%m-%d")

# Take the tail so the window is exactly the right size
full = raw[["t", "y_t"]].iloc[-(SEQ_LEN + PRED_LEN):].reset_index(drop=True)

# Context window fed to the model
series = full.iloc[:SEQ_LEN][["t", "y_t"]].copy().reset_index(drop=True)
series["text"] = ""

# Ground truth for evaluation
gt_vals = full.iloc[SEQ_LEN:]["y_t"].values.astype(np.float32)
```

## Fetching Live Data (Yahoo Finance)

For live market data, use `yfinance`:

```python
import yfinance as yf
from datetime import datetime, timedelta

TICKER = "USO"
CONTEXT_LEN = 64

end = datetime.today()
start = end - timedelta(days=int(CONTEXT_LEN * 2.5))
raw = yf.download(TICKER, start=start.strftime("%Y-%m-%d"),
                  end=end.strftime("%Y-%m-%d"), progress=False)
raw = raw[["Close"]].dropna().tail(CONTEXT_LEN).reset_index()
raw.columns = ["t", "y_t"]
raw["t"] = pd.to_datetime(raw["t"]).dt.strftime("%Y-%m-%d")
raw["text"] = ""
```

## Batch Inference Data

For batch runs, place CSVs in a directory and use `list_data_files`:

```python
from migaseval import list_data_files

csv_files = list_data_files("path/to/csv_dir")
```

Each CSV should follow the same `t`, `y_t` (and optionally `text`) format.

## Downloading the Subset Dataset

The repo includes a download script for the evaluation subset:

```bash
# Download CSVs
uv run python -m migaseval.scripts.download_data --dataset subset --csvs

# Download pre-computed summaries (for offline runs)
uv run python -m migaseval.scripts.download_data --dataset subset --summaries
```

## Common Pitfalls

- **Date format**: Must be `YYYY-MM-DD` strings. `pd.to_datetime(col).dt.strftime("%Y-%m-%d")` fixes most formats.
- **Missing values**: Drop NaNs before feeding to the pipeline. The model does not handle gaps.
- **Column names**: Must be exactly `t` and `y_t` — not `date`, `value`, `close`, etc.
- **Data type**: `y_t` must be numeric (float). Strings or object types will fail silently.
