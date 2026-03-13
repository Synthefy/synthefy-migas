# %% [markdown]
# # Bring Your Own Data
# 
# This notebook shows how to run Migas-1.5 on **your own data** end-to-end:
# 
# 1. **Fetch time series data** — download daily price history from Yahoo Finance.
# 2. **Prepare a text summary** — learn the two-part `FACTUAL SUMMARY` + `PREDICTIVE SIGNALS` format the model expects. A sample summary is provided so the notebook runs out of the box; an optional section shows how to auto-generate one with an LLM.
# 3. **Forecast** — compare Chronos-2 (text-free baseline) against Migas-1.5 (text-conditioned).
# 4. **Counterfactual exploration** — rewrite the predictive signals and watch the forecast shift, demonstrating the text-conditioned time series forecasting.
# 
# **Requirements:** Install the package from the repo root (`uv sync`). Section 3 (LLM summary generation) optionally requires an OpenAI or Anthropic API key.
# 
# **See also:** [Migas-1.5 Inference Quick Start](migas-1.5-inference-quickstart.ipynb) · [Counterfactual Scenarios](migas-1.5-counterfactual-scenarios.ipynb)

# %%
import warnings

warnings.filterwarnings("ignore", message="IProgress not found")

import os
import sys
from textwrap import dedent

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yfinance as yf

from migaseval import MigasPipeline
from migaseval.model.inference_utils import evaluate_chronos_quantiles

sys.path.insert(0, "..")
from scripts.counterfactual_utils import (composite_trend_score,
                                          extract_factual, extract_predictive,
                                          linear_slope, splice_summary)
from scripts.plotting_utils import (COLORS, _draw_forecast_region,
                                    apply_migas_style, plot_forecast_single)
from scripts.summary_utils import generate_summary

apply_migas_style()

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = MigasPipeline.from_pretrained(
    "Synthefy/migas-1.5", device=device, text_embedder_device=device
)
print(f"Using device: {device}")

# %% [markdown]
# ## 1. Get Time Series Data
# 
# Below we fetch daily closing prices for Gold (GLD) from Yahoo Finance and keep the last `SEQ_LEN + PRED_LEN` trading days. The first `SEQ_LEN` days form the **context window** fed to the model; the remaining `PRED_LEN` days are held out as **ground truth** so we can measure forecast accuracy.
# 
# **Why GLD?** Gold responds cleanly to macro text (Fed policy, inflation, risk-off flows), making the text-conditioning effect easy to demonstrate.
# 
# You can swap `TICKER` for any Yahoo Finance symbol: stocks (`AAPL`, `MSFT`), ETFs (`SPY`, `GLD`), or futures (`CL=F` for crude oil).

# %%

TICKER = "GLD"   # Gold ETF — moderate vol, clean macro narrative; swap freely (see note above)
SEQ_LEN = 64    # context length in days (32–384; shorter windows show clearer local trends)
PRED_LEN = 16   # forecast horizon in days

raw = yf.download(TICKER, period="2y", auto_adjust=True, progress=False)["Close"]
raw = raw.dropna().squeeze()

# Fetch context + prediction window so we have ground truth for evaluation
full = raw.iloc[-(SEQ_LEN + PRED_LEN):].reset_index()
full.columns = ["t", "y_t"]
full["t"] = full["t"].dt.strftime("%Y-%m-%d")
full["text"] = ""

# Context window fed to the model
series = full.iloc[:SEQ_LEN].copy()

# Ground truth for the forecast horizon
gt_vals = full.iloc[SEQ_LEN:]["y_t"].values.astype(np.float32)

print(
    f"Context window : {len(series)} days  ({series['t'].iloc[0]} → {series['t'].iloc[-1]})"
)
print(
    f"Forecast window: {len(gt_vals)} days  ({full['t'].iloc[SEQ_LEN]} → {full['t'].iloc[-1]})"
)
print(f"Context price range : ${series['y_t'].min():.2f} – ${series['y_t'].max():.2f}")
series.tail()

# %% [markdown]
# ## 2. Understanding the Summary Format
# 
# Migas-1.5 accepts **text summary** alongside the time series. The summary must follow a two-part structure:
# 
# | Section | Purpose |
# |---------|---------|
# | `FACTUAL SUMMARY` | What already happened — price action, key events, macro drivers |
# | `PREDICTIVE SIGNALS` | Forward-looking interpretation — analyst outlook, catalysts, risks |
# 
# This is the format produced by Migas-1.5's internal `ContextSummarizer` (which calls an LLM over per-timestep news text). The model was trained to condition on this structure, so deviating significantly from it will reduce text impact.
# 
# Below is a sample GLD summary. **To generate a fresh one from real headlines, see Section 3.**

# %%
# Pre-computed sample summary — illustrates the required format.
# Replace this or regenerate it in Section 3 to match your actual data window.
summary = """\
FACTUAL SUMMARY:
GLD has experienced a strong uptrend from November 2025 through January 2026, surging from ~372 to nearly 495 before a sharp pullback to ~445 on January 30, 2026, followed by consolidation in the 427-468 range through mid-February. Key drivers include billionaire hedge fund accumulation (Ken Griffin, Israel Englander), safe-haven demand amid geopolitical tensions and Trump administration policy uncertainty, weakening USD, and central bank buying. Technical support remains in place despite recent volatility, with institutional inflows offsetting prior outflows and multiple analysts citing bullish gold fundamentals for 2026.

PREDICTIVE SIGNALS:
GLD exhibits a bullish medium-term bias supported by ongoing safe-haven demand and institutional conviction, though the sharp January peak followed by consolidation suggests potential for further sideways to higher trading before the next directional move. Near-term momentum appears tentatively positive with recent recovery from lows, but elevated volatility and risk of a deeper pullback remain given the magnitude of the January advance and consolidation pattern, with key risks from USD strength or policy shifts.
"""

print(summary)

# %% [markdown]
# ## 3. (Optional) Generate a Summary with an LLM
#
# A good summary requires **per-day news aligned to the context window**, not just a handful
# of recent headlines. This section generates a `FACTUAL SUMMARY` + `PREDICTIVE SIGNALS` using
# an LLM:
#
# - **Anthropic (recommended)** — Claude uses its built-in web search tool to autonomously
#   find news and market events for the exact date range, then summarizes them. No extra API
#   key required beyond your Anthropic key.
# - **OpenAI / vLLM** — generates a summary from price data only (no web search available).
#
# **Required environment variables:**
# - `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` — required to call the LLM.
#
# If no LLM key is found the section is skipped and the pre-computed summary is kept.

# %%
LLM_PROVIDER = "anthropic"  # "anthropic" | "openai"
LLM_API_KEY  = os.getenv({"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}[LLM_PROVIDER])
LLM_BASE_URL = os.getenv("VLLM_BASE_URL")   # None → use provider default
LLM_MODEL    = os.getenv("VLLM_MODEL")       # None → use provider default

# %%
if not LLM_API_KEY:
    print("Skipping LLM summary generation — no LLM API key found.")
    print("Using the pre-computed summary.")
else:
    summary, news_digest = generate_summary(
        TICKER, series, PRED_LEN,
        llm_provider=LLM_PROVIDER,
        llm_api_key=LLM_API_KEY,
        llm_base_url=LLM_BASE_URL,
        llm_model=LLM_MODEL,
        return_news=True,
    )
    if news_digest:
        print("\n" + "=" * 60)
        print("NEWS DIGEST (raw web search findings)")
        print("=" * 60)
        print(news_digest)

# %% [markdown]
# ## 4. Forecast: Chronos-2 Baseline vs. Migas-1.5
# 
# We run two forecasts on the same numerical context:
# 
# - **Chronos-2** — Migas's own internal Chronos backbone, before text fusion.
# - **Migas-1.5** — same Chronos base, fused with the text summary above.
# 
# Both share the **exact same normalization path and Chronos call** — so the gap
# between the two lines is the pure text conditioning effect.
# 
# **If the pre-written summary doesn't match your actual data window**, text can
# steer the forecast in the wrong direction (Migas MAPE > Chronos MAPE). Run
# Section 3 to generate a window-aligned summary and close that gap.

# %%
context_vals = series["y_t"].values.astype(np.float32)

# Chronos-2 runs in float16 (max representable: 65,504).
# Assets like BTC (~$85k-$109k) overflow during denormalization → NaN.
# We pre-scale to a safe range and invert after forecasting.
_pmax = float(context_vals.max())
PRICE_SCALE = float(10 ** np.floor(np.log10(_pmax) - 1)) if _pmax > 1000 else 1.0
if PRICE_SCALE > 1.0:
    print(
        f"Auto-scaling prices ÷{PRICE_SCALE:.0f} for float16 safety "
        f"(max price {_pmax:.0f} > float16 limit 65,504)"
    )

series_scaled = series.copy()
series_scaled["y_t"] = series_scaled["y_t"] / PRICE_SCALE
gt_scaled = gt_vals / PRICE_SCALE  # scaled ground truth (used internally, kept for reference)
context_tensor = (
    torch.tensor(series_scaled["y_t"].values, dtype=torch.float32)
    .unsqueeze(0)
    .unsqueeze(-1)
    .to(device)  # (1, SEQ_LEN, 1)
)

# Migas-1.5 + its internal Chronos-2 baseline — same normalization path, one call
migas_fc, chronos_fc_raw = pipeline.predict_from_dataframe(
    series_scaled, pred_len=PRED_LEN, seq_len=SEQ_LEN,
    summaries=[summary], return_univariate=True,
)
migas_fc   = migas_fc * PRICE_SCALE    # back to original units
chronos_fc = chronos_fc_raw * PRICE_SCALE  # the exact base Migas used before text fusion

# Chronos-2 uncertainty band (10th / 90th percentile) — standalone call for shading only
chronos_q = evaluate_chronos_quantiles(
    context_tensor, PRED_LEN, device=device, chronos_device=device,
    quantile_levels=[0.1, 0.9],
)
chronos_lo = chronos_q["0.1"][0] * PRICE_SCALE
chronos_hi = chronos_q["0.9"][0] * PRICE_SCALE

print(
    f"Chronos-2 forecast : {chronos_fc.shape}  range [{chronos_fc.min():.0f}, {chronos_fc.max():.0f}]"
    f"  (80% band: [{chronos_lo.min():.0f}, {chronos_hi.max():.0f}])"
)
print(
    f"Migas-1.5 forecast : {migas_fc.shape}  range [{migas_fc.min():.0f}, {migas_fc.max():.0f}]"
)
print("Note: Chronos-2 line is Migas's own internal baseline. Gap = text conditioning effect.")

# %%
t_ctx  = pd.to_datetime(full["t"].values[:SEQ_LEN])
t_pred = pd.to_datetime(list(full["t"].values[SEQ_LEN - 1:]))
last_val = float(context_vals[-1])

fig, ax = plot_forecast_single(
    context_vals,
    gt_vals,
    {"Chronos-2": chronos_fc, "Migas-1.5": migas_fc},
    SEQ_LEN,
    PRED_LEN,
    title=f"Chronos-2 (text-free) vs. Migas-1.5 (text-conditioned) — {TICKER}",
    figsize=(11, 4),
    show_metrics=True,
    text_summary=summary,
    timestamps=full["t"].values,
)
# Add Chronos-2 uncertainty band (10th–90th percentile)
ax.fill_between(
    t_pred,
    np.concatenate([[last_val], chronos_lo]),
    np.concatenate([[last_val], chronos_hi]),
    alpha=0.15,
    color=COLORS["Chronos-2"],
    label="Chronos-2 80% interval",
    zorder=2,
)
ax.legend(fontsize=8, handlelength=1.6, labelspacing=0.35, borderpad=0.45)
ax.set_ylabel(f"{TICKER} Price (USD)", color="#566573")
plt.show()

print(f"Chronos-2 slope  : {linear_slope(chronos_fc):+.5f}")
print(f"Migas-1.5 slope  : {linear_slope(migas_fc):+.5f}")
print(f"Slope difference : {linear_slope(migas_fc) - linear_slope(chronos_fc):+.5f}")

# %% [markdown]
# ## 5. Counterfactual — Rewrite the Narrative
# 
# Here is the core idea behind Migas-1.5: **the numerical input is identical across all
# runs below — only the text changes.**
# 
# We keep the **factual section** unchanged (what already happened doesn't change) and
# replace only the **predictive signals** with a bullish or bearish outlook.
# If the model truly integrates text with time series, the forecast should shift
# in the direction of the new narrative.

# %%
print("Factual section (unchanged across all scenarios):\n")
print(extract_factual(summary))
print("\nOriginal predictive section:\n")
print(extract_predictive(summary))

# %%
bullish_predictive = dedent("""
    PREDICTIVE SIGNALS:
    Gold is entering an extreme upside regime: the Fed has unexpectedly pivoted to an
    aggressive rate-cut path following a sharp deterioration in labor-market data, real
    yields are collapsing, and the dollar is in free fall. Central banks are accelerating
    reserve diversification away from Treasuries, ETF inflows are surging to multi-year
    highs, and retail safe-haven demand is spiking. The path of least resistance is sharply
    higher, with a rapid continuation move and further record highs far more likely than
    any meaningful pullback over the forecast window.
""").strip()

bearish_predictive = dedent("""
    PREDICTIVE SIGNALS:
    Gold is entering an extreme downside regime after a sudden hawkish pivot: the Fed has
    signaled rates will stay higher for longer following a stronger-than-expected CPI print,
    real yields are surging to multi-year highs, and the dollar is strengthening sharply.
    ETF outflows have accelerated, speculative long positions are being unwound, and risk
    appetite has improved dramatically, reducing safe-haven demand. The highest-probability
    path over the forecast window is a rapid, sustained decline as rising yields crush the
    non-yielding metal with little prospect of near-term recovery.
""").strip()

bullish_summary = splice_summary(summary, bullish_predictive)
bearish_summary = splice_summary(summary, bearish_predictive)

# %%
bullish_fc = (
    pipeline.predict_from_dataframe(
        series_scaled, pred_len=PRED_LEN, seq_len=SEQ_LEN, summaries=[bullish_summary]
    )
    * PRICE_SCALE
)
bearish_fc = (
    pipeline.predict_from_dataframe(
        series_scaled, pred_len=PRED_LEN, seq_len=SEQ_LEN, summaries=[bearish_summary]
    )
    * PRICE_SCALE
)

# %%
BULLISH_COLOR = "#2EAD6D"
BEARISH_COLOR = "#C0392B"

t_ctx  = pd.to_datetime(full["t"].values[:SEQ_LEN])
t_pred = pd.to_datetime(list(full["t"].values[SEQ_LEN - 1:]))
last_val = float(context_vals[-1])

fig, ax = plt.subplots(figsize=(11, 5))
_draw_forecast_region(ax, SEQ_LEN, PRED_LEN, boundary=t_pred[0], boundary_end=t_pred[-1])

ax.plot(
    t_ctx,
    context_vals,
    color=COLORS["historical"],
    lw=2.0,
    label="Historical",
    solid_capstyle="round",
)
ax.plot(
    t_pred,
    np.concatenate([[last_val], gt_vals]),
    color=COLORS["ground_truth"],
    lw=2.2,
    label="Ground Truth",
    solid_capstyle="round",
)
ax.plot(
    t_pred,
    np.concatenate([[last_val], migas_fc]),
    color=COLORS["Migas-1.5"],
    lw=2.0,
    ls="--",
    alpha=0.85,
    label="Original (Migas-1.5)",
    solid_capstyle="round",
)
ax.plot(
    t_pred,
    np.concatenate([[last_val], bullish_fc]),
    color=BULLISH_COLOR,
    lw=2.4,
    label="Bullish scenario",
    solid_capstyle="round",
)
ax.plot(
    t_pred,
    np.concatenate([[last_val], bearish_fc]),
    color=BEARISH_COLOR,
    lw=2.4,
    label="Bearish scenario",
    solid_capstyle="round",
)
ax.fill_between(
    t_pred,
    np.concatenate([[last_val], bearish_fc]),
    np.concatenate([[last_val], bullish_fc]),
    alpha=0.08,
    color="#9B8EC4",
    label="Scenario range",
)

ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
for lbl in ax.get_xticklabels():
    lbl.set_rotation(35)
    lbl.set_ha("right")
ax.set_xlabel("Date", color="#566573")
ax.set_ylabel(f"{TICKER} Price (USD)", color="#566573")
ax.set_title(
    f"Migas-1.5 scenario fan — {TICKER}: the same numbers, three different stories",
    fontsize=11,
    fontweight=600,
)
ax.legend(fontsize=8, handlelength=1.6, labelspacing=0.35, borderpad=0.45)
fig.tight_layout(pad=1.2)
plt.show()

# %%
orig_slope = linear_slope(migas_fc)
bull_slope = linear_slope(bullish_fc)
bear_slope = linear_slope(bearish_fc)

rows = [
    {
        "Scenario": "Chronos-2 (no text)",
        "Slope": f"{linear_slope(chronos_fc):+.5f}",
        "Slope shift vs original": "—",
        "Trend score (↑)": f"{composite_trend_score(chronos_fc, 'up', context_vals):+.3f}",
    },
    {
        "Scenario": "Migas-1.5 original",
        "Slope": f"{orig_slope:+.5f}",
        "Slope shift vs original": "—",
        "Trend score (↑)": f"{composite_trend_score(migas_fc, 'up', context_vals):+.3f}",
    },
    {
        "Scenario": "Bullish",
        "Slope": f"{bull_slope:+.5f}",
        "Slope shift vs original": f"{bull_slope - orig_slope:+.5f}",
        "Trend score (↑)": f"{composite_trend_score(bullish_fc, 'up', context_vals):+.3f}",
    },
    {
        "Scenario": "Bearish",
        "Slope": f"{bear_slope:+.5f}",
        "Slope shift vs original": f"{bear_slope - orig_slope:+.5f}",
        "Trend score (↑)": f"{composite_trend_score(bearish_fc, 'up', context_vals):+.3f}",
    },
]
print(pd.DataFrame(rows).to_string(index=False))

# %% [markdown]
# The table confirms what the plot shows: the bullish narrative steers the slope positive,
# the bearish narrative pulls it negative — all with the same 64 days of price history.
# Compare the Chronos-2 row (no text) with Migas-1.5 to see the baseline text effect.
# 
# ## What's next
# 
# - **Try your own asset** — change `TICKER` and `SEQ_LEN` at the top of Section 1.
# - **Generate a fresh summary** — set your API key and re-run Section 3 for a summary
#   grounded in real recent headlines.
# - **Batch evaluation** — see [Backtest and Metrics](migas-1.5-backtest-and-metrics.ipynb)
#   for rolling-window evaluation with ground truth.

# %%



