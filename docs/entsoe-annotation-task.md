# Task: Build a Text-Annotated Energy Price Time Series from ENTSO-E

## Goal

Produce a single well-annotated CSV file for **one European country or bidding zone** (e.g. Germany-Luxembourg `DE_LU`) covering **2024** that looks like this:

| t | y_t | text |
|---|-----|------|
| 2024-01-01 | 52.3 | Planned maintenance at Ringhals 3 (nuclear, 1063 MW). Unplanned trip at Kraftwerk Moorburg Block A due to cooling fault. |
| 2024-01-02 | 48.1 | No active outages. |
| … | … | … |

- **`t`** — date string `YYYY-MM-DD`
- **`y_t`** — daily average day-ahead electricity price (EUR/MWh)
- **`text`** — free-text summary of REMIT outage events active on that day

This file will be fed directly into **Migas-1.5** as a text-conditioned time series for forecasting. The `text` column is what drives the model's text conditioning — so quality matters.

---

## Background

### What is ENTSO-E?

The European Network of Transmission System Operators for Electricity. Their [Transparency Platform](https://transparency.entsoe.eu/) publishes real-time and historical data on electricity prices, generation, and grid events across Europe.

### What data do we need?

**1. Day-ahead prices (document type A44)**
Hourly EUR/MWh prices published each day for the next day. We resample to daily averages for the model.

**2. REMIT Urgent Market Messages (document type A77)**
Outage notices filed by generators and TSOs whenever a power plant or transmission asset becomes unavailable. Each message contains:
- Asset name (e.g. "Ringhals 3", "Kraftwerk Moorburg Block A")
- Fuel type (nuclear, gas, coal, wind, etc.)
- Event type (planned maintenance vs unplanned trip)
- Start/end datetime of the outage
- **Free-text reason field** — this is the annotation we want

### Why REMIT for text?

REMIT messages are the closest thing to structured news about the power grid. An unplanned trip of a 1,400 MW nuclear plant is a real market event — it moves prices and it has a text description. This gives us grounded, time-stamped text annotations that are directly causally related to the price series.

---

## What Has Already Been Done

The repo branch `entsoe-bulk-download` on `github.com/Synthefy/synthefy-migas` contains:

- **`scripts/fetch_entsoe.py`** — fetches prices (A44) + REMIT messages (A77/A78) via HTTPS and joins them into a single CSV
- **`scripts/entsoe_bulk_download.py`** — alternative bulk downloader using the `entsoe-py` library

### What `fetch_entsoe.py` already does

```
entsoe_prices_DE_LU.csv    — raw hourly prices
entsoe_remit_DE_LU.csv     — raw REMIT messages with reason_text, asset_name, fuel_type
entsoe_joined_DE_LU.csv    — hourly prices joined with active outage events
```

The join is hourly. **Your job is to aggregate this to daily and build the clean `t / y_t / text` CSV.**

---

## Your Task

### Step 1 — Set up and get an API token

1. Register at [transparency.entsoe.eu](https://transparency.entsoe.eu/)
2. Go to **My Account Settings → Web API Security Token → Generate**
3. Clone the repo and check out the branch:
   ```bash
   git clone git@github.com:Synthefy/synthefy-migas.git
   cd synthefy-migas
   git checkout entsoe-bulk-download
   uv sync
   ```

### Step 2 — Fetch the data

Run `fetch_entsoe.py` for your chosen zone. Start with **DE_LU (Germany-Luxembourg)** — it has the richest REMIT coverage.

```bash
# Jan–Jun (run on one machine)
ENTSOE_API_TOKEN=<your-token> uv run python scripts/fetch_entsoe.py \
  --start 202401010000 --end 202407010000 \
  --output-dir ./entsoe_data_h1

# Jul–Dec (run in parallel on another machine or just run sequentially)
ENTSOE_API_TOKEN=<your-token> uv run python scripts/fetch_entsoe.py \
  --start 202407010000 --end 202412310000 \
  --output-dir ./entsoe_data_h2
```

> **Note on speed:** The REMIT fetch is slow (~1.5–2 hours per half-year) because the API has a 200-record-per-request limit and returns dense data in summer months. The script handles this automatically with adaptive window halving — just let it run. You can split across two machines as shown above to halve the time.

> **Known issue:** A78 (generation unavailability) returns empty data for DE_LU via this endpoint — A77 (which covers both transmission and generation UMMs in practice) is the one with data. This is expected.

After both runs finish, concatenate:

```python
import pandas as pd

prices = pd.concat([
    pd.read_csv("entsoe_data_h1/entsoe_prices_DE_LU.csv"),
    pd.read_csv("entsoe_data_h2/entsoe_prices_DE_LU.csv"),
]).drop_duplicates("timestamp_utc").sort_values("timestamp_utc")

remit = pd.concat([
    pd.read_csv("entsoe_data_h1/entsoe_remit_DE_LU.csv"),
    pd.read_csv("entsoe_data_h2/entsoe_remit_DE_LU.csv"),
]).drop_duplicates().sort_values("event_start")
```

### Step 3 — Build the daily annotated CSV

This is the core task. Convert the hourly joined data into a daily `t / y_t / text` file.

```python
import pandas as pd

# Load joined hourly data (or re-join from raw files above)
joined = pd.concat([
    pd.read_csv("entsoe_data_h1/entsoe_joined_DE_LU.csv"),
    pd.read_csv("entsoe_data_h2/entsoe_joined_DE_LU.csv"),
])
joined["timestamp_utc"] = pd.to_datetime(joined["timestamp_utc"], utc=True)
joined["date"] = joined["timestamp_utc"].dt.date

# Daily average price
daily_price = joined.groupby("date")["price_eur_mwh"].mean().reset_index()
daily_price.columns = ["t", "y_t"]

# Daily text: aggregate all outage reason texts for that day
def build_daily_text(group):
    texts = group["outage_texts"].dropna()
    texts = texts[texts != ""]
    if texts.empty:
        return ""
    # Deduplicate — same outage spans many hours
    unique = list(dict.fromkeys("; ".join(texts).split("; ")))
    unique = [t.strip() for t in unique if t.strip()]
    return ". ".join(unique[:10])  # cap at 10 distinct events per day

daily_text = joined.groupby("date").apply(build_daily_text).reset_index()
daily_text.columns = ["t", "text"]

# Merge
result = daily_price.merge(daily_text, on="t")
result["t"] = result["t"].astype(str)
result = result.sort_values("t").reset_index(drop=True)

result.to_csv("de_lu_prices_annotated_2024.csv", index=False)
print(result.head())
print(f"\n{len(result)} days, {(result.text != '').sum()} with text annotations")
```

### Step 4 — Validate the output

Check the annotation quality before handing off to the model:

```python
# Spot-check a few days
print(result[result.text != ""].sample(5)[["t", "y_t", "text"]].to_string())

# Coverage stats
print(f"Days with annotations: {(result.text != '').sum()} / {len(result)}")
print(f"Avg price: {result.y_t.mean():.2f} EUR/MWh")
print(f"Price range: {result.y_t.min():.2f} – {result.y_t.max():.2f}")

# Flag days with very long text (might need trimming)
result["text_len"] = result.text.str.len()
print(result.nlargest(5, "text_len")[["t", "text_len", "text"]])
```

A good annotated day looks like:
```
t: 2024-01-15
y_t: 67.4
text: Yearly maintenance at Ringhals 3 (nuclear, SE3). Planned outage at
      Moorburg Block A (lignite, DE). Reduced capacity at EVM gas unit.
```

### Step 5 — Save to the right place

The final file goes in:
```
data/entsoe/de_lu_prices_annotated_2024.csv
```

This path is where the inference notebooks expect custom data. The file is then used like any other Migas-1.5 input — point `DATA_PATH` at it in the quickstart notebook.

---

## Acceptance Criteria

- [ ] Full year 2024 (at minimum Jan–Dec, daily rows)
- [ ] `t` column is `YYYY-MM-DD` string
- [ ] `y_t` column is daily mean day-ahead price in EUR/MWh, no NaNs
- [ ] `text` column populated for days with active REMIT events (expect ~60–80% coverage for DE_LU)
- [ ] Text is human-readable, not raw semicolon-joined noise — clean it up if needed
- [ ] File saved at `data/entsoe/de_lu_prices_annotated_2024.csv`
- [ ] Committed to the `entsoe-bulk-download` branch

---

## Bidding Zone Reference

If you want to try a different country instead of DE_LU:

| Label | Zone | EIC Code |
|-------|------|----------|
| DE_LU | Germany-Luxembourg | `10Y1001A1001A82H` |
| FR | France | `10YFR-RTE------C` |
| ES | Spain | `10YES-REE------0` |
| NL | Netherlands | `10YNL----------L` |
| BE | Belgium | `10YBE----------2` |
| AT | Austria | `10YAT-APG------L` |
| PL | Poland | `10YPL-AREA-----S` |
| FI | Finland | `10YFI-1--------U` |

France and Spain tend to have clean, well-labelled REMIT data and are good alternatives to DE_LU.

---

## Questions / Context

Reach out to Raimi with questions. The relevant code is all on the `entsoe-bulk-download` branch. The model this data feeds into is documented in `notebooks/pyfiles/migas-1.5-inference-quickstart.py`.
