# TE Electricity: Text pipeline for electricity price series

Prepares a **text column** for the 5 Trading Economics electricity price series (Germany, Spain, France, UK, Italy) by combining TE news with external news where TE has no coverage. The text is used as metadata by the TTFM model: it should describe **state** (recent levels, drivers) and **forward-looking signals** (forecasts, expectations). Media Cloud often returns only titles + URLs, which is not useful; **Firecrawl** or **Parallel Web** provide full article content or long excerpts and are recommended for substantive text.

**Outputs** go under `data/te_electricity/`. Raw numeric CSVs are copied from `data/te_commodities/raw/` so the electricity dataset is self-contained.

## Stems (series)

| Stem            | Country  |
|-----------------|----------|
| `deuelepri_com` | Germany  |
| `espelepri_com` | Spain    |
| `fraelepri_com` | France   |
| `gbrelepri_com` | UK       |
| `itaelepri_com` | Italy    |

## Prerequisites

- **Raw historical CSVs** for the 5 electricity series in `data/te_commodities/raw/` (from `te_fetch_historical.py`).
- **TE news** (optional): `data/te_commodities/te_news_by_symbol/{stem}.json` for each stem. If missing or empty, only MC news will be used.
- **Media Cloud API key**: `MEDIACLOUD_API_KEY` or `MC_API_KEY`. Sign up at https://search.mediacloud.org
- **mediacloud** package: `uv sync --extra mediacloud`

## Pipeline (from repo root)

### 1. Prepare raw data

Copy the 5 electricity raw CSVs into `data/te_electricity/raw/`:

```bash
uv run python scripts/te_electricity/prepare_raw.py
```

### 2. Fetch external news (choose one)

You need **substantive text** per article (not just titles + URLs) so the daily summarizer can produce useful state + forward-looking summaries for TTFM.

**Option A — Firecrawl (recommended)**  
Search + scrape: returns full page markdown so each item has real content.

- Set `FIRECRAWL_API_KEY`. Install: `uv sync --extra firecrawl`
- Fetches only **gap dates** (dates with no TE news) by default. Use `--no-gaps` to fetch all raw dates.

```bash
uv run python scripts/te_electricity/firecrawl_fetch_electricity.py
# Optional: one stem, date range
uv run python scripts/te_electricity/firecrawl_fetch_electricity.py --stem espelepri_com --d1 2024-01-01 --d2 2024-12-31
```

Output: `data/te_electricity/fc_news/{stem}.json`. Use `--mc-dir data/te_electricity/fc_news` in step 3.

**Option B — Parallel Web**  
Search API returns LLM-optimized excerpts (long) per result; no separate scrape.

- Set `PARALLEL_API_KEY`. Install: `uv sync --extra parallel`

```bash
uv run python scripts/te_electricity/parallel_fetch_electricity.py
uv run python scripts/te_electricity/parallel_fetch_electricity.py --stem espelepri_com --d1 2024-01-01
```

Output: `data/te_electricity/parallel_news/{stem}.json`. Use `--mc-dir data/te_electricity/parallel_news` in step 3.

**Option C — Media Cloud (often off-topic, titles only)**  
Uses curated queries in `data/te_electricity/electricity_queries.json`. Rate limit ~2 req/min.

```bash
uv run python scripts/te_electricity/mc_fetch_electricity.py --d1 2018-10-01 --d2 2026-02-13
```

Output: `data/te_electricity/mc_news/{stem}.json`. You must run step 2b to get article bodies.

### 2b. Enrich MC news with full article text (optional, only if using Media Cloud)

MC items initially have only title + URL. To get more text, fetch each URL and extract the article body with [trafilatura](https://trafilatura.readthedocs.io/). Writes to `data/te_electricity/mc_news_enriched/`. Slow (one request per story + `--delay`); use `--stem` to run one series at a time if needed.

```bash
uv run python scripts/te_electricity/enrich_mc_news.py
# Or one stem only (e.g. UK):
uv run python scripts/te_electricity/enrich_mc_news.py --stem gbrelepri_com
```

Then use the enriched dir when merging: `--mc-dir data/te_electricity/mc_news_enriched`.

Requires: `trafilatura` (included with `uv sync --extra mediacloud`).

### 3. Merge TE + external news (external fills gaps only)

For each stem, keeps all TE news and adds external news only for dates that have no TE text (gap dates). Writes to `data/te_electricity/merged_news/`. Point `--mc-dir` at whichever source you used (Firecrawl, Parallel, or MC).

```bash
uv run python scripts/te_electricity/merge_mc_into_te_gaps.py
# With Firecrawl output:
uv run python scripts/te_electricity/merge_mc_into_te_gaps.py --mc-dir data/te_electricity/fc_news
# With Parallel output:
uv run python scripts/te_electricity/merge_mc_into_te_gaps.py --mc-dir data/te_electricity/parallel_news
# With enriched MC (if you used Media Cloud):
uv run python scripts/te_electricity/merge_mc_into_te_gaps.py --mc-dir data/te_electricity/mc_news_enriched
```

If raw CSVs are still only in `te_commodities/raw`, pass:

```bash
uv run python scripts/te_electricity/merge_mc_into_te_gaps.py --raw-dir data/te_commodities/raw
```

### 4. Daily summaries (LLM)

Run the shared daily summarizer on the merged news. Requires an LLM server (e.g. vLLM).

```bash
uv run python scripts/run_daily_summaries_by_symbol.py \
  --news-dir data/te_electricity/merged_news \
  --output-dir data/te_electricity/te_daily_summaries
```

### 5. Merge text into raw CSVs

Produces `data/te_electricity/{stem}_with_text.csv` for each stem (columns: `t`, `y_t`, `text`).

```bash
uv run python scripts/merge_all_te_text.py \
  --summaries-dir data/te_electricity/te_daily_summaries \
  --data-dir data/te_electricity
```

## Data layout

```
data/te_electricity/
├── electricity_queries.json   # MC search queries (stem -> name, query)
├── raw/                        # Raw CSVs (from prepare_raw.py)
│   ├── deuelepri_com.csv
│   ├── espelepri_com.csv
│   ├── fraelepri_com.csv
│   ├── gbrelepri_com.csv
│   └── itaelepri_com.csv
├── mc_news/                    # MC per-stem JSONs (optional; from mc_fetch_electricity.py)
├── mc_news_enriched/          # Optional: MC + full article text (from enrich_mc_news.py)
├── fc_news/                    # Firecrawl per-stem JSONs (recommended; from firecrawl_fetch_electricity.py)
├── parallel_news/              # Parallel Web per-stem JSONs (from parallel_fetch_electricity.py)
├── merged_news/                # TE + external gap-filled (from merge_mc_into_te_gaps.py)
├── te_daily_summaries/         # One JSON per stem (from run_daily_summaries_by_symbol)
├── deuelepri_com_with_text.csv # Final evals (from merge_all_te_text)
├── espelepri_com_with_text.csv
├── ...
```

## Optional: edit queries

Edit `data/te_electricity/electricity_queries.json`:

- **`query`**: Media Cloud syntax (AND/OR, double quotes). These were tried on Media Cloud and often went off-topic, so MC is not recommended.
- **`firecrawl_query`**: Plain search string for Firecrawl. Include **"news article"** to get article-style pages with prose (not just OMIE dashboards/PDFs). Used by `firecrawl_fetch_electricity.py`; only results with enough body text (see `--min-body-chars`) are kept so the model gets useful forecast context.
- **`parallel_objective`**: Natural-language objective for Parallel Web search. Used by `parallel_fetch_electricity.py` when set.

Tightening around **wholesale / day-ahead / spot** and exchange names (OMIE, EPEX, EEX, N2EX, GME) helps avoid off-topic hits.

## Eval

Use `--datasets_dir data/te_electricity` (or point to the directory containing the `*_with_text.csv` files) when running evaluation scripts.
