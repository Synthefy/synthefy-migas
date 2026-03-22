# ENTSO-E / Nord Pool Outage Text Analysis

Analysis of European energy market outage text (REMIT UMMs) and their relationship to hourly electricity prices across Nordic bidding zones (2016--2025).

## Key Findings

1. **Individual outage text has weak predictive signal.** After deseasonalizing (removing daily price cycle), most outage reason buckets show no statistically significant price impact at +6h. Five buckets are significant but all negative (price drops), which is counterintuitive.

2. **Concurrent outages matter more than outage type.** When 6+ outages overlap in a zone, price impact is 3-5x larger and doesn't recover within 24h. Solo outages barely move prices.

3. **Neither event count nor total MW offline is a strong predictor** of price direction (r < 0.25 for all zones). The effect is statistically significant due to sample size (25k+ events) but the effect size is small.

4. **Zone context is critical.** The same outage type has opposite effects in different zones. FI (Finland) shows extreme swings; NO1/NO2 (Norway, hydro-dominated) absorb outages with minimal price impact.

5. **Seasonality is predictable.** Icing clusters Nov-Dec (55%), planned maintenance peaks May-Sep, transformer failures spike in December.

## Data Sources

| Source | What | Access |
|---|---|---|
| [ENTSO-E Transparency API](https://transparency.entsoe.eu/) | Hourly day-ahead prices per bidding zone | Free, requires API key |
| [Nord Pool UMM API](https://ummapi.nordpoolgroup.com) | REMIT Urgent Market Messages with human-written outage text | Free, no API key |

**Important:** ENTSO-E API provides NO free-text outage descriptions. Only Nord Pool UMM has human-written text, and only for Nordic/Baltic zones (NO, SE, FI, DK, EE, LT, LV, IE). Central European zones (DE, FR, ES, PL) are not available through Nord Pool.

### Coverage

- **Zones:** NO1, NO2, FI, DK1, DK2, SE1, SE3
- **Period:** 2016--2025
- **Events:** ~29,000 UMM events matched to ~566,000 hourly price rows
- **LLM-classified buckets:** 4,882 unique reason texts classified into 26 semantic categories

## Quick Start

### 1. Setup

```bash
# Clone and install
git clone <repo-url>
cd entsoe-outage-analysis
uv sync

# Configure API keys
cp .env.example .env
# Edit .env: add your ENTSOE_TOKEN (required for prices)
# ANTHROPIC_API_KEY is only needed for LLM bucket classification
```

### 2. Fetch data (full pipeline)

The full pipeline fetches ENTSO-E hourly prices and Nord Pool UMM outage text for 7 zones x 10 years, then combines them into training Parquet files.

```bash
# Run the full pipeline (takes ~1-2 hours, resumable)
ENTSOE_API_TOKEN=your-token uv run python scripts/build_training_data.py
```

This produces:
```
data/entsoe_raw/<zone>_<year>/    # Raw data per zone/year
  umm_raw_<zone>_<year>.json      #   Raw UMM API response
  umm_parsed_<zone>_<year>.csv    #   Parsed outage events (flat CSV)
  entsoe_prices_<zone>.csv        #   Hourly day-ahead prices
data/entsoe/<zone>_<year>.parquet # Daily annotated dataset (t, y_t, text)
data/entsoe/entsoe_training.parquet # Combined training dataset
```

Or fetch individual zones/years:

```bash
# Step 1: Fetch hourly prices from ENTSO-E
uv run python scripts/fetch_entsoe.py \
  --start 202401010000 --end 202412310000 \
  --zone 10YNO-1--------2 --zone-label NO1 \
  --output-dir ./data/entsoe_raw/NO1_2024 --prices-only

# Step 2: Fetch UMM text from Nord Pool + build daily Parquet
uv run python scripts/fetch_nordpool_umm.py \
  --zone NO1 --year 2024 \
  --prices-csv data/entsoe_raw/NO1_2024/entsoe_prices_NO1.csv
```

### 3. Run analysis

Analysis scripts read from `data/entsoe_examples_new/` (parsed UMM CSVs + hourly prices). After fetching, copy or symlink to that path.

```bash
# Hourly price impact (deseasonalized boxplots, trajectories)
uv run python scripts/entsoe_hourly_impact.py

# LLM bucket classification (requires ANTHROPIC_API_KEY in .env)
uv run python scripts/classify_buckets_llm.py

# Text bucket analysis (uses LLM buckets if available)
uv run python scripts/entsoe_text_buckets.py

# Deep analysis: cascade, MW, recovery, seasonality, signal audit
uv run python scripts/entsoe_bucket_deep_analysis.py

# Presentation plots from daily Parquet files
uv run python scripts/entsoe_findings_plots.py

# Annotated time series visualizations
uv run python scripts/entsoe_annotated_timeseries.py
uv run python scripts/entsoe_weekly_annotated.py
```

All plots are saved to `figures/`.

## Scripts

All scripts run with `uv run python scripts/<name>.py` from the repo root.

### Data Pipeline

| Script | Purpose | Input | Output |
|---|---|---|---|
| `fetch_entsoe.py` | Fetch hourly day-ahead prices from ENTSO-E Transparency API | ENTSO-E API token, zone EIC code, date range | `entsoe_prices_<zone>.csv` |
| `fetch_nordpool_umm.py` | Fetch REMIT UMM messages from Nord Pool (free, no key) and build daily annotated dataset | Zone label, year, optional prices CSV | `umm_raw_*.json`, `umm_parsed_*.csv`, `<zone>_<year>.parquet` |
| `build_training_data.py` | Full pipeline orchestrator: runs fetch_entsoe + fetch_nordpool_umm for 7 zones x 10 years, skips completed, combines into training parquet | `ENTSOE_API_TOKEN` env var | `data/entsoe/entsoe_training.parquet` |
| `entsoe_bulk_download.py` | Bulk download CSVs from ENTSO-E (alternative to API fetch) | ENTSO-E token | `entsoe_data/<dataset>/<zone>.csv` |

### Analysis

| Script | Purpose | Key Outputs |
|---|---|---|
| `entsoe_hourly_impact.py` | Hourly price impact after outages, deseasonalized, rolling avg and single-point | Boxplots by type/zone; trajectories by reason, fuel, severity |
| `entsoe_text_buckets.py` | Price impact grouped by LLM-classified outage reason buckets | Heatmaps, trajectories, boxplots (all zones + per zone) |
| `entsoe_bucket_deep_analysis.py` | Deep analysis: bucket x MW, bucket x zone, recovery time, cascade, seasonality, signal audit | 10+ plots, statistical tests, correlation comparison |
| `entsoe_findings_plots.py` | Presentation-ready summary plots from daily data | Coverage heatmap, signal by zone, fuel breakdown |
| `classify_buckets_llm.py` | Uses Claude API to classify outage reason texts into semantic buckets | `data/reason_text_buckets.json` (4,882 entries) |

### Visualization

| Script | Purpose |
|---|---|
| `entsoe_annotated_timeseries.py` | Multi-month price charts with outage event annotations and arrows |
| `entsoe_weekly_annotated.py` | 7-day zoomed plots showing ALL outages with price overlay and unavailable MW |

## Pipeline Architecture

```
ENTSO-E Transparency API          Nord Pool UMM API
(hourly prices, API key)          (outage text, free)
         │                                │
    fetch_entsoe.py              fetch_nordpool_umm.py
         │                                │
    entsoe_prices_*.csv          umm_parsed_*.csv + *.parquet
         │                                │
         └──────────┬─────────────────────┘
                    │
           build_training_data.py
                    │
           entsoe_training.parquet
            (t, y_t, text, series_id)
                    │
        ┌───────────┼───────────────┐
        │           │               │
  hourly_impact  text_buckets  bucket_deep_analysis
        │           │               │
     figures/    figures/         figures/
```

### Data Formats

**Training Parquet** (`data/entsoe/<zone>_<year>.parquet`):

| Column | Type | Description |
|---|---|---|
| `t` | str | Date (YYYY-MM-DD) |
| `y_t` | float | Daily average day-ahead price (EUR/MWh) |
| `text` | str | Concatenated outage descriptions active on that day |

**Parsed UMM CSV** (`umm_parsed_<zone>_<year>.csv`):

| Column | Type | Description |
|---|---|---|
| `event_start` | datetime | Outage start (UTC) |
| `event_end` | datetime | Outage end (UTC) |
| `unavailability_type` | str | "Planned" or "Unplanned" |
| `reason_text` | str | Human-written outage reason |
| `remarks` | str | Additional notes |
| `asset_name` | str | Power plant / unit name |
| `fuel_type` | str | Generation fuel (Nuclear, Hydro, etc.) |
| `unavailable_mw` | float | Capacity offline (MW) |

## Figures

### Main Analysis (`figures/`)

| Figure | Description |
|---|---|
| `entsoe_bucket_x_mw_heatmap.png` | Bucket x MW severity: does outage size matter differently per type? |
| `entsoe_bucket_x_zone_heatmap.png` | Bucket x zone: which outage types hit which zones hardest? |
| `entsoe_bucket_recovery.png` | Peak price impact and recovery time by bucket |
| `entsoe_cascade_effect.png` | Price trajectory by number of concurrent outages (all zones) |
| `entsoe_cascade_per_zone.png` | Same, per zone |
| `entsoe_cascade_total_mw.png` | Price trajectory by total MW offline in zone |
| `entsoe_cascade_total_mw_per_zone.png` | Same, per zone |
| `entsoe_bucket_seasonality.png` | Monthly distribution of outage types |
| `entsoe_bucket_signal_audit.png` | Which buckets carry statistically significant signal? |
| `entsoe_bucket_heatmap.png` | Price impact heatmap: bucket x time horizon |
| `entsoe_bucket_trajectories.png` | Hourly price trajectories per bucket (top 12) |
| `entsoe_bucket_boxplot_6h.png` | Distribution of price change at +6h per bucket |
| `entsoe_hourly_boxplot_type.png` | Unplanned vs planned price impact at multiple horizons |
| `entsoe_hourly_boxplot_per_zone.png` | Same, broken down by zone |
| `entsoe_coverage_heatmap.png` | Annotation coverage across zones and years |

### Annotated Time Series (`figures/weekly/`)

Weekly 7-day plots for FI Nov-Dec 2022 showing every outage event annotated on the price curve with unavailable MW in the bottom panel.

## Methodology

### Deseasonalization

Hourly electricity prices follow a strong daily cycle (high during day, low at night). To isolate the outage effect:

1. Compute average price per hour-of-day per zone (the "seasonal profile")
2. For each outage event at hour H0, the expected price change at hour H is: `profile[H] - profile[H0]`
3. Deseasonalized change = actual change - expected seasonal change

### Cascade Concurrency

For each outage event, we count how many other outages are active in the same zone at event start, using the declared `event_start` and `event_end` timestamps from UMM data. An event is "active" if `event_start <= t < event_end`.

### LLM Bucket Classification

Used Claude (Sonnet) to classify 7,082 unique outage reason texts into 26 semantic buckets (e.g., "Turbine failure", "Planned maintenance", "Icing/cold weather"). Batches of 200-300 texts, incremental JSON saving, resumable. Achieved 4,882 classifications (69% of unique texts, covering ~77% of events).

## Dependencies

Core:
```
pandas, numpy, matplotlib, scipy, requests, pytz, python-dotenv
```

Optional (for LLM classification only):
```
anthropic
```

Install everything: `uv sync` from repo root. For LLM classification: `uv sync --extra llm`.
