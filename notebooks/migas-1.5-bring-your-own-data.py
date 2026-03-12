# %% [markdown]
# # Bring Your Own Data
#
# This notebook shows how to run Migas-1.5 on **your own data** end-to-end:
#
# 1. **Fetch time series data** — download daily price history from Yahoo Finance (no API key needed).
# 2. **Prepare a text summary** — learn the two-part `FACTUAL SUMMARY` + `PREDICTIVE SIGNALS` format the model expects. A sample summary is provided so the notebook runs out of the box; an optional section shows how to auto-generate one with an LLM.
# 3. **Forecast** — compare Chronos-2 (text-free baseline) against Migas-1.5 (text-conditioned).
# 4. **Counterfactual exploration** — rewrite the predictive signals and watch the forecast shift, demonstrating the model's core novelty: text-conditioned time series forecasting.
#
# **Requirements:** Install the package from the repo root (`uv sync`). Sections 1–4 and the counterfactuals in Section 5 run **without any API keys**. Section 3 (LLM summary generation) optionally requires an OpenAI or Anthropic API key.
#
# **See also:** [Migas-1.5 Inference Quick Start](migas-1.5-inference-quickstart.ipynb) · [Counterfactual Scenarios](migas-1.5-counterfactual-scenarios.ipynb)

# %%
import warnings
from collections import defaultdict

import requests

from migaseval.model.inference_utils import evaluate_chronos_quantiles

warnings.filterwarnings("ignore", message="IProgress not found")

import os
import sys
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yfinance as yf

from migaseval import MigasPipeline

sys.path.insert(0, "..")
from scripts.counterfactual_utils import (composite_trend_score,
                                          extract_factual, extract_predictive,
                                          linear_slope, splice_summary)
from scripts.plotting_utils import (COLORS, _draw_forecast_region,
                                    apply_migas_style, plot_forecast_single)

apply_migas_style()

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = MigasPipeline.from_pretrained(
    "Synthefy/migas-1.5", device=device, text_embedder_device=device
)
print(f"Using device: {device}")

# %% [markdown]
# ## 1. Get Time Series Data
#
# Migas-1.5 accepts context windows between **64 and 384 time steps**. Below we fetch two years of daily closing prices for Gold (GLD) from Yahoo Finance — no API key required — and keep the last `SEQ_LEN + PRED_LEN` trading days. The first `SEQ_LEN` days form the **context window** fed to the model; the remaining `PRED_LEN` days are held out as **ground truth** so we can measure forecast accuracy.
#
# **Why GLD?** Gold has moderate volatility (~8% annualized vs. ~70% for BTC), so the model's median forecast shows a visible directional slope rather than a flat line. It also responds cleanly to macro text (Fed policy, inflation, risk-off flows), making the text-conditioning effect easy to demonstrate.
#
# You can swap `TICKER` for any Yahoo Finance symbol: stocks (`AAPL`, `MSFT`), ETFs (`SPY`, `GLD`), or futures (`CL=F` for crude oil). **Avoid high-volatility crypto** — the median forecast will look flat regardless of text input.

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
# Migas-1.5 accepts an optional **text summary** alongside the time series. The summary must follow a two-part structure:
#
# | Section | Purpose |
# |---------|---------|
# | `FACTUAL SUMMARY` | What already happened — price action, key events, macro drivers |
# | `PREDICTIVE SIGNALS` | Forward-looking interpretation — analyst outlook, catalysts, risks |
#
# This is the format produced by Migas-1.5's internal `ContextSummarizer` (which calls an LLM over per-timestep news text). The model was trained to condition on this structure, so deviating significantly from it will reduce text impact.
#
# Below is a sample GLD summary. It runs without any API calls. **To generate a fresh one from real headlines, see Section 3.**

# %%
# Pre-computed sample summary — illustrates the required format.
# Replace this or regenerate it in Section 3 to match your actual data window.
summary = """\
FACTUAL SUMMARY:
Over the past 32 trading days gold (GLD) advanced steadily from around $230 to $245,
supported by a weakening U.S. dollar, rising geopolitical uncertainty, and persistent
inflation above the Fed's 2% target. Central bank buying — particularly from emerging-
market central banks — provided a durable bid, while real yields dipped as rate-cut
expectations were pulled forward following softer-than-expected jobs data.

PREDICTIVE SIGNALS:
Analysts broadly expect gold to remain well-bid near current levels, with upside risk
if the Fed signals an earlier pivot or geopolitical tensions escalate further. ETF
inflows have turned positive after months of outflows, suggesting renewed institutional
interest. The primary downside risk is a sharp re-pricing of U.S. rate expectations
higher, which could push real yields up and weigh on non-yielding assets like gold.\
"""

print(summary)

# %% [markdown]
# ## 3. (Optional) Generate a Summary with an LLM
#
# A good summary requires **per-day news aligned to the context window**, not just a handful
# of recent headlines. This section uses the **Alpha Vantage News Sentiment API** to fetch
# articles for the exact date range of the context window, groups them by day, and builds a
# structured prompt that mirrors the format Migas-1.5's internal `ContextSummarizer` uses
# (timestamped entries with price values alongside the text). The LLM then distills this into
# `FACTUAL SUMMARY` + `PREDICTIVE SIGNALS`.
#
# **Required environment variables** (set at least one pair):
# - `ALPHA_VANTAGE_API_KEY` — free at alphavantage.co (25 req/day on free tier)
# - `LLM_PROVIDER` + one of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`
#
# If either key is missing the section is skipped and the pre-computed summary is kept.
# Free Alpha Vantage keys support ~2 years of historical news, which covers our 64-day window.

# %%

LLM_PROVIDER = "anthropic"  # "openai" | "anthropic"

_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
# LLM_API_KEY = os.getenv(_KEY_ENV[LLM_PROVIDER])
LLM_API_KEY = "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"
LLM_BASE_URL = os.getenv("VLLM_BASE_URL", None)  # None → use provider default
LLM_MODEL = os.getenv("VLLM_MODEL", None)  # None → use provider default
# ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
ALPHA_VANTAGE_KEY = "6RM2VZCN6CPXJ8B4"


# %%
def call_llm(
    prompt,
    *,
    provider=LLM_PROVIDER,
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    model=LLM_MODEL,
):
    """Call an LLM and return the response text.

    Supports OpenAI-compatible endpoints (including local vLLM: set LLM_BASE_URL
    and keep provider='openai') and the Anthropic Messages API.
    """
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    elif provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model or "claude-haiku-4-5-20251001",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER {provider!r}. Use 'openai' or 'anthropic'."
        )


def _to_av_ticker(ticker):
    """Convert a yfinance ticker to Alpha Vantage News Sentiment format.

    Alpha Vantage uses different conventions:
      - Crypto  : BTC-USD  → CRYPTO:BTC
      - Stocks  : AAPL     → AAPL  (unchanged)
      - Futures : CL=F     → None  (not supported; search topic instead)
    """
    if "-USD" in ticker or "-USDT" in ticker:
        return f"CRYPTO:{ticker.split('-')[0]}"
    if ticker.endswith("=F"):
        return None   # futures — caller should omit tickers param
    return ticker


def fetch_av_news(ticker, start_date, end_date, api_key, limit=200):
    """Fetch news articles from Alpha Vantage News Sentiment API.

    Returns a list of article dicts sorted by publication time.
    Each dict has at minimum: time_published (str YYYYMMDDTHHMMSS), title, summary.
    """
    av_ticker = _to_av_ticker(ticker)
    params = {
        "function": "NEWS_SENTIMENT",
        "time_from": start_date.strftime("%Y%m%dT0000"),
        "time_to":   end_date.strftime("%Y%m%dT2359"),
        "limit":     limit,
        "apikey":    api_key,
    }
    if av_ticker:
        params["tickers"] = av_ticker
    else:
        # Futures: fall back to a broad topic search (energy, commodities)
        params["topics"] = "energy_transportation"

    resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
    data = resp.json()
    if "Information" in data:
        raise RuntimeError(f"Alpha Vantage API error: {data['Information']}")
    return data.get("feed", [])


def align_news_to_dates(articles, dates):
    """Group articles by calendar date and return one text string per date.

    Articles on the same day are concatenated as 'Title. Summary.' pairs,
    separated by ' | '. Days with no coverage get an empty string.
    """
    by_date = defaultdict(list)
    for art in articles:
        pub = art.get("time_published", "")
        if len(pub) < 8:
            continue
        date_key = f"{pub[:4]}-{pub[4:6]}-{pub[6:8]}"
        title = art.get("title", "").strip()
        summary = art.get("summary", "").strip()
        text = f"{title}. {summary}" if summary else title
        if text:
            by_date[date_key].append(text)

    return [" | ".join(by_date.get(d, [])) for d in dates]


def build_context_summarizer_prompt(ticker, dates, prices, per_day_texts, pred_period):
    """Build the same prompt structure as ContextSummarizer._create_prompt.

    Each line: [date] (value: price): text   (or 'No text' if empty)
    This is the exact format Migas-1.5 was trained to summarise.
    """
    lines = []
    for date, price, text in zip(dates, prices, per_day_texts):
        entry_text = text if text else "No text"
        lines.append(f"[{date}] (value: {price:.4f}): {entry_text}")
    combined = "\n---\n".join(lines)

    return f"""\
You are analyzing a time series with text annotations. \
Extract information to help forecast future values for {ticker}.

HISTORICAL DATA:
{combined}

PREDICTION PERIOD: {pred_period}

Provide TWO sections:

SECTION 1 - FACTUAL SUMMARY:
Summarize observed facts, patterns, trends, and key events. (2-3 sentences)

SECTION 2 - PREDICTIVE SIGNALS:
Identify forward-looking directional signals, expectations, and sentiment for
the forecast window. Express momentum and risk in RELATIVE terms only — for
example: "likely to continue higher", "risk of 5-10% pullback", "bullish bias
with upside momentum". Do NOT include absolute price levels, specific support/
resistance numbers, or external analyst price targets — these often refer to a
different instrument or unit (e.g. gold spot $/oz vs. ETF price) and will
mislead the model. (2-3 sentences)

Format:
FACTUAL SUMMARY:
[Your factual summary]

PREDICTIVE SIGNALS:
[Your predictive signals]"""


# %%
_missing = [
    k
    for k, v in [("LLM key", LLM_API_KEY), ("ALPHA_VANTAGE_API_KEY", ALPHA_VANTAGE_KEY)]
    if not v
]
if _missing:
    print(f"Skipping LLM summary generation — missing: {', '.join(_missing)}")
    print("Using the pre-computed summary.")
else:
    start_dt = pd.Timestamp(series["t"].iloc[0])
    end_dt = pd.Timestamp(series["t"].iloc[-1])
    pred_period = f"the {PRED_LEN} days after {series['t'].iloc[-1]}"

    print(
        f"Fetching Alpha Vantage news for {TICKER} "
        f"({start_dt.date()} → {end_dt.date()}) …"
    )
    articles = fetch_av_news(TICKER, start_dt, end_dt, ALPHA_VANTAGE_KEY)
    print(f"  Retrieved {len(articles)} articles")

    dates = series["t"].tolist()
    prices = series["y_t"].tolist()
    per_day_texts = align_news_to_dates(articles, dates)

    days_with_news = sum(1 for t in per_day_texts if t)
    print(f"  Days with at least one article: {days_with_news} / {len(dates)}")

    prompt = build_context_summarizer_prompt(
        TICKER, dates, prices, per_day_texts, pred_period
    )
    print(f"Calling {LLM_PROVIDER} to generate summary …")
    summary = call_llm(prompt)
    print("\nGenerated summary:\n")
    print(summary)

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
t_ctx  = np.arange(SEQ_LEN)
t_pred = np.arange(SEQ_LEN - 1, SEQ_LEN + PRED_LEN)
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

t_ctx = np.arange(SEQ_LEN)
t_pred = np.arange(SEQ_LEN - 1, SEQ_LEN + PRED_LEN)
last_val = float(context_vals[-1])

fig, ax = plt.subplots(figsize=(11, 5))
_draw_forecast_region(ax, SEQ_LEN, PRED_LEN)

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

ax.set_xlabel("Time Step", color="#566573")
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
