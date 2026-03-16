# Run Forecasts with Migas-1.5

When the user asks how to run a forecast, predict, or do inference with Migas-1.5,
follow these instructions.

## Pipeline Setup

```python
import torch
from migaseval import MigasPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = MigasPipeline.from_pretrained(
    "Synthefy/migas-1.5", device=device, text_embedder_device=device
)
```

## Single Series Forecast (DataFrame API)

The simplest way — pass a DataFrame with `t`, `y_t` columns:

```python
# Without text summary (uses per-row text column if present)
forecast = pipeline.predict_from_dataframe(
    series,              # pd.DataFrame with t, y_t (and optionally text) columns
    pred_len=16,         # forecast horizon
    seq_len=64,          # context window (optional, defaults to len(series))
)
# forecast is a 1D np.ndarray of shape (pred_len,)

# With a text summary (overrides per-row text)
forecast = pipeline.predict_from_dataframe(
    series,
    pred_len=16,
    seq_len=64,
    summaries=[summary],  # list of 1 summary string
)
```

## Batch Forecast (Array API)

For multiple series in one call — more efficient than looping:

```python
import numpy as np

# context_batch: shape (B, seq_len), float32
# texts: list of B lists, each containing seq_len strings
context_batch = np.stack([s1_values, s2_values])  # (B, seq_len)
texts = [s1_texts, s2_texts]                       # list of B lists

forecast_batch = pipeline.predict(
    context_batch,
    texts,
    pred_len=16,
)
# forecast_batch: torch.Tensor, shape (B, pred_len, 1)
```

## Chronos-2 Baseline Comparison

Chronos-2 is Migas's internal backbone without text conditioning. Compare the
two to isolate the text effect:

```python
from migaseval.model.inference_utils import evaluate_chronos, evaluate_chronos_quantiles

# Migas-1.5 + its internal Chronos baseline — same normalization path
migas_fc, chronos_fc = pipeline.predict_from_dataframe(
    series, pred_len=PRED_LEN, seq_len=SEQ_LEN,
    summaries=[summary], return_univariate=True,
)
# migas_fc: shape (pred_len,) — text-conditioned
# chronos_fc: shape (pred_len,) — text-free baseline

# Chronos-2 uncertainty band (standalone call)
context_tensor = torch.tensor(context_vals).unsqueeze(0).unsqueeze(-1).to(device)
chronos_q = evaluate_chronos_quantiles(
    context_tensor, PRED_LEN, device=device, chronos_device=device,
    quantile_levels=[0.1, 0.9],
)
chronos_lo = chronos_q["0.1"][0]  # 10th percentile
chronos_hi = chronos_q["0.9"][0]  # 90th percentile
```

## Counterfactual Forecasts

Run the same numerical context with different text summaries to see the
text-conditioning effect:

```python
from migaseval.counterfactual_utils import splice_summary

bullish_summary = splice_summary(summary, bullish_predictive)
bearish_summary = splice_summary(summary, bearish_predictive)

fc_original = pipeline.predict_from_dataframe(series, pred_len=PRED_LEN, summaries=[summary])
fc_bullish  = pipeline.predict_from_dataframe(series, pred_len=PRED_LEN, summaries=[bullish_summary])
fc_bearish  = pipeline.predict_from_dataframe(series, pred_len=PRED_LEN, summaries=[bearish_summary])
```

## Rolling-Window Backtest

Slide a context window through the series to evaluate forecast quality:

```python
from migaseval.model.inference_utils import evaluate_chronos

seq_len = 64
pred_len = 16
stride = pred_len  # non-overlapping; use stride=1 for fully rolling

migas_forecasts = []
chronos_forecasts = []
ground_truths = []

for i in range((len(df) - seq_len - pred_len) // stride + 1):
    start = i * stride
    end = start + seq_len
    gt = df["y_t"].values[end : end + pred_len]
    if len(gt) < pred_len:
        break

    df_ctx = df.iloc[start:end]
    migas_pred = pipeline.predict_from_dataframe(df_ctx, pred_len=pred_len)

    ctx_tensor = torch.tensor(df_ctx["y_t"].values.astype(np.float32)).reshape(1, -1, 1)
    chronos_pred = evaluate_chronos(ctx_tensor, pred_len=pred_len, device=device)
    chronos_pred = chronos_pred[0, :, 0].detach().cpu().numpy()

    migas_forecasts.append(migas_pred)
    chronos_forecasts.append(chronos_pred)
    ground_truths.append(gt.astype(np.float32))
```

## Typical Parameter Values

| Parameter | Typical | Notes |
|-----------|---------|-------|
| `seq_len` | 64–384 | Shorter = clearer local trends; longer = more history |
| `pred_len` | 8–64 | Forecast horizon in steps |
| `stride` | `pred_len` (non-overlapping) or `1` (fully rolling) | For backtests |

## Metrics

Standard metrics for evaluating forecast quality:

```python
def compute_backtest_metrics(preds, ground_truths, last_ctx):
    errors = preds - ground_truths
    mae = np.mean(np.abs(errors), axis=1)
    mse = np.mean(errors ** 2, axis=1)
    mape = np.mean(np.abs(errors) / (np.abs(ground_truths) + 1e-8), axis=1) * 100
    # Directional accuracy (step 1)
    pred_dir = np.sign(preds[:, 0] - last_ctx)
    gt_dir = np.sign(ground_truths[:, 0] - last_ctx)
    valid = gt_dir != 0
    dir_acc = (pred_dir[valid] == gt_dir[valid]).mean() * 100
    return {"MAE": np.mean(mae), "MSE": np.mean(mse),
            "MAPE%": np.mean(mape), "Dir.Acc%": dir_acc}
```

## Trend Metrics (for Counterfactuals)

```python
from migaseval.counterfactual_utils import (
    linear_slope,
    composite_trend_score,
    endpoint_change,
    monotonicity,
)

slope = linear_slope(forecast)                              # least-squares slope
score = composite_trend_score(forecast, "up", context_vals) # weighted composite
ep_chg = endpoint_change(forecast)                          # relative first-to-last change
mono = monotonicity(forecast, "up")                         # fraction of steps in target direction
```
