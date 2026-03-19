# Migas-1.5 — Claude Code Guide

This repo contains **Migas-1.5**, a text-conditioned time-series forecasting model.
It fuses text summaries (news, analyst commentary) with numerical time series to
produce forecasts that respond to narrative context.

## Before writing any code

1. **Read the matching reference notebook** from `notebooks/pyfiles/` — these are
   the canonical usage patterns. Do not deviate from them.
2. **Read the matching skill file(s)** from `.claude/skills/` — they explain *why*
   the rules exist and what breaks when you deviate.

| Reference notebook | When to read |
|---|---|
| `notebooks/pyfiles/migas-1.5-inference-quickstart.py` | Any forecast, summary generation, or counterfactual task |
| `notebooks/pyfiles/migas-1.5-counterfactual-scenarios.py` | Counterfactual / sentiment scenario analysis |
| `notebooks/pyfiles/migas-1.5-backtest-and-metrics.py` | Backtesting or metric evaluation |
| `notebooks/pyfiles/migas-1.5-batch-inference.py` | Batch / multi-series inference |

## API key setup

Summary generation uses Claude with web search to find real news. **Requires `ANTHROPIC_API_KEY`.**

If the user doesn't have it set in the environment or `.env`:
1. **Ask them to provide it** — web-search-powered summaries produce significantly
   better forecasts than price-only fallbacks.
2. Help them create `.env` with `ANTHROPIC_API_KEY=sk-ant-...`
3. Code loads it via `dotenv.load_dotenv()`.

## Running scripts

- **Always use `uv run python`** to execute scripts (e.g. `uv run python scripts/foo.py`).
- If the virtualenv doesn't exist or deps are missing, run **`uv sync`** first.

## Key conventions

- Data columns: `t` (YYYY-MM-DD string), `y_t` (float), `text` (str, optional)
- Summaries must have exactly two sections: `FACTUAL SUMMARY:` and `PREDICTIVE SIGNALS:`
- PREDICTIVE SIGNALS: **relative terms only** — no absolute price targets or specific
  support/resistance numbers
- `generate_summary()` returns `list[str]` (default `n_summaries=9`); passing multiple
  summaries to `predict_from_dataframe` triggers ensemble averaging for stability
- `splice_summary`, `extract_factual`, `extract_predictive` all accept `str | list[str]`
- Counterfactual predictive text must start with `PREDICTIVE SIGNALS:`
- Forecast dates: `pd.bdate_range(start=last_date, periods=PRED_LEN + 1)` — business days
- Always `apply_migas_style()` before any plot
- Always prepend `last_val` to forecast arrays: `np.concatenate([[last_val], forecast])`
- Use `format_date_axis` from `migaseval.plotting_utils` for date x-axes
- LLM calls use `call_llm()` from `migaseval.summary_utils` — not raw API calls

## Project layout

```
src/migaseval/          core package (pipeline, model, evaluation, plotting)
notebooks/              Jupyter notebooks + pyfiles/ (.py exports)
scripts/                data download and preprocessing utilities
data/                   sample CSVs (JPM, oil, energy)
figures/                saved plot outputs
.claude/skills/         skill files with frontmatter triggers
```
