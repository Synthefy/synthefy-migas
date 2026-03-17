# Migas-1.5 — Claude Code Guide

This repo contains **Migas-1.5**, a text-conditioned time-series forecasting model.
It fuses text summaries (news, analyst commentary) with numerical time series to
produce forecasts that respond to narrative context.

> **STOP. Before writing any code, you MUST complete steps 1 and 2 below.
> Skipping them silently degrades forecast quality and breaks the pipeline.**

## Quick-start example

If the user asks something like *"forecast gold prices"* or *"can you predict GLD
under different sentiments?"*, here is the complete pattern to follow.

**Before running:** check for `ANTHROPIC_API_KEY` in the environment or `.env` file.
If missing, **ask the user to provide their Anthropic API key** — it is required for
web-search-powered summary generation, which produces much better forecasts than
the price-only fallback. Set it in `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

**Minimal working script** (GLD gold ETF, bullish vs bearish counterfactual):

```python
import os, warnings, torch, numpy as np, pandas as pd, yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message="IProgress not found")
load_dotenv()

from migaseval import MigasPipeline
from migaseval.counterfactual_utils import splice_summary
from migaseval.plotting_utils import (
    COLORS, _draw_forecast_region, apply_migas_style, format_date_axis,
)
from migaseval.summary_utils import generate_summary

apply_migas_style()

# ── Config ────────────────────────────────────────────────────────────────────
TICKER      = "GLD"
SERIES_NAME = "SPDR Gold Shares (GLD)"
CONTEXT_LEN = 64
PRED_LEN    = 16
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ── Load model ────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = MigasPipeline.from_pretrained("Synthefy/migas-1.5", device=device)

# ── Fetch data via yfinance ───────────────────────────────────────────────────
end   = datetime.today()
start = end - timedelta(days=int(CONTEXT_LEN * 2.5))
raw   = yf.download(TICKER, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), progress=False)
raw   = raw[["Close"]].dropna().tail(CONTEXT_LEN).reset_index()
raw.columns = ["t", "y_t"]
raw["t"]    = pd.to_datetime(raw["t"]).dt.strftime("%Y-%m-%d")
raw["text"] = ""

context_vals = raw["y_t"].values.astype(np.float32)

# ── Generate summary (requires ANTHROPIC_API_KEY) ────────────────────────────
summary, news_digest = generate_summary(
    SERIES_NAME, raw, PRED_LEN,
    llm_provider="anthropic",
    llm_api_key=ANTHROPIC_API_KEY,
    return_news=True,
)

# ── Counterfactual scenarios ──────────────────────────────────────────────────
bullish_predictive = """\
PREDICTIVE SIGNALS: Safe-haven demand is accelerating as geopolitical tensions \
escalate and central banks continue aggressive gold purchases, suggesting the \
current rally is likely to extend with upside momentum over the forecast window. \
Institutional flows into gold ETFs are at multi-year highs, reinforcing the \
bullish bias."""

bearish_predictive = """\
PREDICTIVE SIGNALS: Rising real yields and a strengthening dollar are creating \
significant headwinds for gold, suggesting a pullback of 5-10% is likely over \
the forecast window. Institutional positioning is increasingly crowded long, \
raising the risk of a sharp unwind if risk appetite returns to equities."""

bullish_summary = splice_summary(summary, bullish_predictive)
bearish_summary = splice_summary(summary, bearish_predictive)

# ── Run forecasts ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

fc_original = pipeline.predict_from_dataframe(raw, pred_len=PRED_LEN, summaries=[summary])
fc_bullish  = pipeline.predict_from_dataframe(raw, pred_len=PRED_LEN, summaries=[bullish_summary])
fc_bearish  = pipeline.predict_from_dataframe(raw, pred_len=PRED_LEN, summaries=[bearish_summary])

# ── Plot ──────────────────────────────────────────────────────────────────────
t_ctx  = pd.to_datetime(raw["t"].values)
last_d = pd.to_datetime(raw["t"].iloc[-1])
t_pred = pd.bdate_range(start=last_d, periods=PRED_LEN + 1)
last_v = float(context_vals[-1])

fig, ax = plt.subplots(figsize=(11, 5))
_draw_forecast_region(ax, CONTEXT_LEN, PRED_LEN, boundary=t_pred[0], boundary_end=t_pred[-1])

ax.plot(t_ctx, context_vals, color=COLORS["historical"], lw=2.0,
        label="Historical", solid_capstyle="round")
ax.plot(t_pred, np.concatenate([[last_v], fc_original]),
        color=COLORS["Migas-1.5"], lw=2.2, ls="--", alpha=0.85,
        label="Original forecast", solid_capstyle="round")
ax.plot(t_pred, np.concatenate([[last_v], fc_bullish]),
        color="#2EAD6D", lw=2.4, label="Bullish scenario", solid_capstyle="round")
ax.plot(t_pred, np.concatenate([[last_v], fc_bearish]),
        color="#C0392B", lw=2.4, label="Bearish scenario", solid_capstyle="round")
ax.fill_between(t_pred,
    np.concatenate([[last_v], fc_bearish]),
    np.concatenate([[last_v], fc_bullish]),
    alpha=0.08, color="#9B8EC4", label="Scenario range")

format_date_axis(ax)
ax.set_xlabel("Date"); ax.set_ylabel("Price ($)")
ax.set_title(f"Migas-1.5 · {SERIES_NAME}: bullish vs bearish scenarios", fontsize=11, fontweight=600)
ax.legend(fontsize=8)
fig.tight_layout(pad=1.2)
fig.savefig("figures/gld_scenario_forecast.png", dpi=150, bbox_inches="tight")
plt.show()
```

## Step 1 — Read the reference notebooks (ALWAYS first)

The canonical usage patterns live in `notebooks/pyfiles/`. Read the relevant
file(s) before writing anything:

| File | When to read |
|------|-------------|
| `notebooks/pyfiles/migas-1.5-inference-quickstart.py` | Any forecast, summary generation, or counterfactual task |
| `notebooks/pyfiles/migas-1.5-counterfactual-scenarios.py` | Counterfactual / sentiment scenario analysis |
| `notebooks/pyfiles/migas-1.5-backtest-and-metrics.py` | Backtesting or metric evaluation |
| `notebooks/pyfiles/migas-1.5-batch-inference.py` | Batch / multi-series inference |

**Key patterns the pyfiles establish (do not deviate):**

- Summary generation always uses `generate_summary(series_name, series, pred_len, llm_provider, llm_api_key, return_news=True)` → returns `(summary, news_digest)` tuple.
- LLM-generated counterfactual signals use `call_llm(prompt, provider, api_key, ...)` from `migaseval.summary_utils` — not a raw API call.
- Counterfactual predictive text passed to `splice_summary` must start with `PREDICTIVE SIGNALS:` (the function handles stripping bold markers).
- Forecast dates use `pd.bdate_range(start=last_date, periods=PRED_LEN + 1)` — business days, length `PRED_LEN + 1` to match the prepended `last_val`.
- Always import `format_date_axis` from `migaseval.plotting_utils` for date x-axes.
- `plot_scenario_comparison` and `display_text_comparison` are available from `migaseval.counterfactual_utils`.

## Step 2 — Read the matching skill file(s)

| Task | Skill file | When to use |
|------|-----------|-------------|
| Prepare or format data | `.claude/skills/prepare-data.md` | Any time data is loaded, fetched, or structured for the pipeline |
| Generate a text summary | `.claude/skills/generate-summary.md` | Any time a FACTUAL SUMMARY / PREDICTIVE SIGNALS string is written or modified |
| Run a forecast or backtest | `.claude/skills/run-forecast.md` | Any time `MigasPipeline` is used or forecast code is written |
| Plot forecasts | `.claude/skills/plotting.md` | Any time a forecast chart is created |

**Rule:** read the matching skill file **before** writing code. The pyfiles show
*how* to call the API; the skill files explain *why* the rules exist and what
breaks when you deviate from them. You need both.

## API key setup

Summary generation uses Claude with web search to find real news for the context
window. **This requires an Anthropic API key.**

If the user does not have `ANTHROPIC_API_KEY` set in the environment or `.env`:
1. **Ask them to provide it** — explain that it enables web-search-powered summaries
   which produce significantly better forecasts than price-only fallbacks.
2. Help them create a `.env` file with `ANTHROPIC_API_KEY=sk-ant-...`
3. The code loads it via `dotenv.load_dotenv()`.

Without an API key, the pipeline falls back to price-data-only summaries (no news
context), which limits the text-conditioning effect.

## Key conventions

- Data columns: `t` (YYYY-MM-DD string), `y_t` (float), `text` (str, optional)
- Summaries must have exactly two sections: `FACTUAL SUMMARY:` and `PREDICTIVE SIGNALS:`
- PREDICTIVE SIGNALS must use **relative terms only** — no absolute price targets or
  specific support/resistance numbers
- Always call `apply_migas_style()` before any plot
- Always prepend `last_val` to forecast arrays when plotting:
  `np.concatenate([[last_val], forecast])`
- Use `uv run python` to run scripts

## Project layout

```
src/migaseval/          core package (pipeline, model, evaluation, plotting)
notebooks/              Jupyter notebooks + pyfiles/ (.py exports)
scripts/                data download and preprocessing utilities
data/                   sample CSVs (JPM, oil, energy)
figures/                saved plot outputs
.claude/skills/         skill files — read before coding
```
