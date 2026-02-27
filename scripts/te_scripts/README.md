# TE country indicators pipeline

Scripts to explore Trading Economics news for 20 enterprise countries, fetch historical indicator series, and produce TTFM eval datasets (CSV with columns `t`, `y_t`, `text`).

## Prerequisites

- Raw news: monthly JSONs in `data/te_commodities/te_news_raw_old/` (from `te_fetch_news_to_summaries.py` without country filter).
- Country config: `data/te_countries/te_countries.json` (20 countries and name aliases).
- API key: `TRADING_ECONOMICS_API_KEY` or `TE_API_KEY` for fetch and optional `--with-forecast` in exploration.

## Pipeline

1. **Explore** (no API required unless `--with-forecast`):
   ```bash
   uv run python scripts/te_scripts/explore_country_indicator_news.py
   ```
   Writes `data/te_countries/exploration_ranked.csv` with (country, symbol, news_count, unique_dates, days_in_range, daily_coverage_fraction, ...) sorted by daily news presence. Use `--top N` to limit rows; `--min-days-in-range` and `--min-unique-dates` filter which pairs are included.

2. **Fetch historical** (requires API key):
   ```bash
   uv run python scripts/te_scripts/fetch_country_indicator_historical.py --input-csv data/te_countries/exploration_ranked.csv --top 50
   ```
   Calls TE `/historical/ticker/{ticker}/{d1}` per symbol; writes `data/te_countries/raw/{country}_{symbol}.csv` (t, y_t, frequency) and `raw/manifest.json`.

3. **Build text and merge** (raw text or vLLM summaries):

   **Option A – raw text** (no LLM):
   ```bash
   uv run python scripts/te_scripts/build_country_indicator_text.py
   ```
   Loads raw news, aggregates by period, merges into raw CSVs; writes `data/te_countries/with_text/{stem}.csv`.

   **Option B – vLLM daily summaries** (recommended for better text quality):
   - Build news JSON for create_daily_summaries (one file per stem, format: date + llm_summary per article):
     ```bash
     uv run python scripts/te_scripts/build_news_for_summaries.py
     ```
     Writes `data/te_countries/news_for_summaries/{stem}.json`.
   - Run vLLM daily summarization (requires LLM server, e.g. on port 8004):
     ```bash
     uv run python scripts/te_scripts/run_daily_summaries_te_countries.py --llm-url http://localhost:8004/v1 --llm-model openai/gpt-oss-120b
     ```
     Writes `data/te_countries/te_daily_summaries/{stem}.json` (date, daily_summary per day).
   - Merge summaries into raw CSVs (aggregates daily summaries by period to match series frequency):
     ```bash
     uv run python scripts/te_scripts/build_country_indicator_text.py --summaries-dir data/te_countries/te_daily_summaries --output-dir data/te_countries/with_text
     ```
   Use `--output-dir data/te_countries/with_text_summarized` if you want to keep both raw-text and summarized outputs.

## Daily presence and plots (visual date-range choice)

4. **Build daily presence CSVs** (t, text_present per calendar day):
   ```bash
   uv run python scripts/te_scripts/build_daily_presence.py
   ```
   Reads `exploration_ranked.csv` and raw news; writes `data/te_countries/daily_presence/{stem}.csv` with one row per day from date_min to date_max and `text_present` True/False.

5. **Plot daily presence** (timeline + optional 30-day rolling share):
   ```bash
   uv run python scripts/te_scripts/plot_daily_presence.py
   ```
   Reads `daily_presence/*.csv` and saves `data/te_countries/daily_presence_plots/{stem}.png`. Use these plots to visually pick date ranges with consistent text. Options: `--rolling 30`, `--stem United_States_SPX`.

## Output layout

- `data/te_countries/exploration_ranked.csv` – ranked (country, symbol) by text presence.
- `data/te_countries/raw/` – numeric CSVs and `manifest.json`.
- `data/te_countries/with_text/` – final CSVs for `LateFusionDataset` (t, y_t, text).
- `data/te_countries/daily_presence/` – per-series CSVs with `t`, `text_present` (bool).
- `data/te_countries/daily_presence_plots/` – PNG timelines for visual period selection.
- `data/te_countries/news_for_summaries/` – per-stem JSON for create_daily_summaries (date, llm_summary per article).
- `data/te_countries/te_daily_summaries/` – per-stem daily summaries from vLLM (date, daily_summary per day).
