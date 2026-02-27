# TE country-indicator pipeline — scripts and commands

**Assumption:** You already have **global** TE raw news (no country filter) as monthly JSONs in `data/te_commodities/te_news_raw_old/` (e.g. from `scripts/te_fetch_news_to_summaries.py` without `--countries`). Config: `data/te_countries/te_countries.json` (20 countries + aliases).

---

## 1. Explore (rank country+symbol by daily news presence)

**Script:** `explore_country_indicator_news.py`  
**What:** Loads raw news, keeps 20 countries, groups by (country, symbol), computes stats and ranks by `daily_coverage_fraction`. Writes ranked CSV.

```bash
uv run python scripts/te_scripts/explore_country_indicator_news.py
# Optional: --min-days-in-range 365 --min-unique-dates 450 --top 55
```

**Output:** `data/te_countries/exploration_ranked.csv`

---

## 2. Optional: daily presence CSVs + plots (visual period choice)

**Scripts:** `build_daily_presence.py`, `plot_daily_presence.py`  
**What:** For each row in ranked CSV, build a CSV with one row per calendar day and `text_present` (bool). Then plot timelines (and optional rolling share) so you can pick date ranges with consistent text.

```bash
uv run python scripts/te_scripts/build_daily_presence.py
uv run python scripts/te_scripts/plot_daily_presence.py
```

**Output:** `data/te_countries/daily_presence/{stem}.csv`, `data/te_countries/daily_presence_plots/{stem}.png`

---

## 3. Fetch historical numeric series (TE API)

**Script:** `fetch_country_indicator_historical.py`  
**What:** Reads ranked CSV (country, symbol), calls TE `/historical/ticker/{symbol}/{d1}` per symbol, writes one CSV per (country, symbol) with `t`, `y_t`, `frequency` and `manifest.json`.

**Requires:** `TRADING_ECONOMICS_API_KEY` or `TE_API_KEY`

```bash
export TRADING_ECONOMICS_API_KEY="your_key"
uv run python scripts/te_scripts/fetch_country_indicator_historical.py --input-csv data/te_countries/exploration_ranked.csv
# Optional: --top 50 --d1 2015-01-01 --d2 2026-02-28
```

**Output:** `data/te_countries/raw/{stem}.csv`, `data/te_countries/raw/manifest.json`

---

## 4. Merge text into numeric CSVs (TTFM-ready t, y_t, text)

**Option A — raw text (no LLM)**  
**Script:** `build_country_indicator_text.py`  
**What:** Loads raw news, aggregates by period (week/month/year) per series frequency, merges into raw CSVs, trims leading empty text.

```bash
uv run python scripts/te_scripts/build_country_indicator_text.py
```

**Output:** `data/te_countries/with_text/{stem}.csv`

---

**Option B — vLLM daily summaries (recommended)**

**4a. Build news JSON for summarizer**  
**Script:** `build_news_for_summaries.py`  
**What:** From raw news + manifest, writes one JSON per stem with list of `{ date, llm_summary, title, url }` (one per article) in the format expected by `create_daily_summaries.py`.

```bash
uv run python scripts/te_scripts/build_news_for_summaries.py
```

**Output:** `data/te_countries/news_for_summaries/{stem}.json`

**4b. Run vLLM daily summarization**  
**Script:** `run_daily_summaries_te_countries.py`  
**What:** Calls `scripts/run_daily_summaries_by_symbol.py` with news dir = `news_for_summaries` and output dir = `te_daily_summaries`. One vLLM summary per day per stem.

**Requires:** vLLM (or compatible) server running.

```bash
uv run python scripts/te_scripts/run_daily_summaries_te_countries.py \
  --llm-url http://localhost:8006/v1 \
  --llm-model openai/gpt-oss-120b
```

**Output:** `data/te_countries/te_daily_summaries/{stem}.json`

**4c. Merge summaries into raw CSVs**  
**Script:** `build_country_indicator_text.py` (with `--summaries-dir`)  
**What:** Loads daily summaries per stem, aggregates by period to match series frequency, merges into raw CSVs.

```bash
uv run python scripts/te_scripts/build_country_indicator_text.py \
  --summaries-dir data/te_countries/te_daily_summaries \
  --output-dir data/te_countries/with_text
```

**Output:** `data/te_countries/with_text/{stem}.csv` (t, y_t, text with summarized content)

---

## Scripts overview

| Script | Purpose |
|--------|--------|
| `explore_country_indicator_news.py` | Rank (country, symbol) by daily news presence → exploration_ranked.csv |
| `build_daily_presence.py` | Per-stem CSV: t, text_present (for plotting) |
| `plot_daily_presence.py` | Plot daily presence timelines → PNGs |
| `fetch_country_indicator_historical.py` | TE API: fetch numeric series → raw/*.csv + manifest.json |
| `build_news_for_summaries.py` | Raw news → news_for_summaries/*.json (vLLM input) |
| `run_daily_summaries_te_countries.py` | Run create_daily_summaries for each stem → te_daily_summaries/*.json |
| `build_country_indicator_text.py` | Merge text (raw or summaries) into raw CSVs → with_text/*.csv |

---

## Slice by period spec (most-annotated segments)

**Script:** `slice_by_period_spec.py`  
**What:** Reads `data/te_countries/period_spec.json` (country, symbol, periods). For each, slices the corresponding `with_text/*.csv` by the given date range(s) and writes to `with_text_most_annotated/`. Multiple periods for the same country+symbol produce multiple CSVs (e.g. `Commodity_KC1_2017-2018.csv`, `Commodity_KC1_2021-onwards.csv`).

```bash
uv run python scripts/te_scripts/slice_by_period_spec.py
```

**Spec format:** `"weekly full"` / `"fine"` = full file; `">=2018"` = from 2018-01-01; `">=2020-08-01"` = from that date; `">=2020 <=2024"` = 2020–2024 inclusive; `">=2017 <=2018; >=2021"` = two segments. Edit `data/te_countries/period_spec.json` to change ranges.

**Output:** `data/te_countries/with_text_most_annotated/*.csv`

---

## Minimal pipeline (no vLLM)

1. Explore → 2. Fetch historical → 3. Build text (raw) → (optional) slice by period spec → use `with_text/*.csv` or `with_text_most_annotated/*.csv` for evals.

## Full pipeline (with vLLM summaries)

1. Explore → 2. (optional) Daily presence + plot → 3. Fetch historical → 4a. Build news for summaries → 4b. Run vLLM daily summaries → 4c. Merge summaries → (optional) slice by period spec → use `with_text_most_annotated/*.csv` for evals.
