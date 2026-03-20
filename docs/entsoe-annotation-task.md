# Task: Build a Text-Annotated Energy Price Time Series from ENTSO-E

## Goal

Produce a set of well-annotated Parquet files — one per country/zone per year — that look like this:

| t | y_t | text |
|---|-----|------|
| 2024-01-01 | 52.3 | Planned maintenance at Ringhals 3 (nuclear, 1063 MW). Unplanned trip at Kraftwerk Moorburg Block A due to cooling fault. |
| 2024-01-02 | 48.1 | No active outages. |
| … | … | … |

- **`t`** — date string `YYYY-MM-DD`
- **`y_t`** — daily average day-ahead electricity price (EUR/MWh)
- **`text`** — free-text summary of REMIT outage events active on that day

The target is **~10,000 annotated daily rows** to start, which is roughly 5 zones × 5–6 years each. More is better. These files will be used to train Migas-1.5 on energy price forecasting with text conditioning.

**Use Parquet throughout, not CSV.** Parquet is faster to read, smaller on disk, and preserves dtypes. All intermediate and final files should be `.parquet`.

---

## What Has Already Been Done ✓

The repo branch `entsoe-bulk-download` on `github.com/Synthefy/synthefy-migas` already contains:

- **`scripts/fetch_entsoe.py`** — fetches day-ahead prices (A44) + REMIT outage messages (A77) via HTTPS and joins them hourly. Supports `--start`, `--end`, `--zone`, `--zone-label`, `--output-dir` flags.
- **`scripts/entsoe_bulk_download.py`** — alternative bulk downloader (less relevant for this task)
- **`entsoe_data/`** — day-ahead prices, load, generation for 12 European zones covering 2024 (already fetched, in the repo)
- **`entsoe_prices_DE_LU.csv`** — hourly prices for Germany-Luxembourg, 2024
- **`entsoe_joined_DE_LU.csv`** — prices joined with REMIT ⚠️ **0 REMIT rows** — the REMIT fetch for this file was not completed

**What is NOT done:** the REMIT text fetch, daily aggregation, annotation quality validation, and the training dataset.

---

## Phase 1 — Validate One Zone First

**Do this before anything else.** Confirm the REMIT annotations are actually predictive of price changes. If they're not, we need a different text source.

### Step 1 — Setup

Clone the repo and check out the branch:

```bash
git clone git@github.com:Synthefy/synthefy-migas.git
cd synthefy-migas
git checkout entsoe-bulk-download
uv sync
```

API token:
```
a447c244-ce1a-4568-bd84-cb601805204a
```

### Step 2 — Fetch DE_LU for 2024

Start with **Germany-Luxembourg** — it has the richest REMIT coverage.

```bash
ENTSOE_API_TOKEN=a447c244-ce1a-4568-bd84-cb601805204a uv run python scripts/fetch_entsoe.py \
  --start 202401010000 --end 202412310000 \
  --zone 10Y1001A1001A82H --zone-label DE_LU \
  --output-dir ./entsoe_raw/DE_LU_2024
```

> **This takes ~3–4 hours.** The REMIT API has a 200-record-per-request limit and returns dense data in summer. The script handles this automatically with adaptive window halving — just let it run.

> **Fetch one zone at a time.** The REMIT API returns data for the entire European grid per request, not just your zone — fetching multiple zones in parallel hits limits fast.

> **Known issue:** A78 (generation unavailability) returns empty data — A77 covers both in practice. This is expected.

This produces:
```
entsoe_raw/DE_LU_2024/entsoe_prices_DE_LU.csv
entsoe_raw/DE_LU_2024/entsoe_remit_DE_LU.csv
entsoe_raw/DE_LU_2024/entsoe_joined_DE_LU.csv
```

### Step 3 — Build the daily annotated Parquet

```python
import pandas as pd

joined = pd.read_csv("entsoe_raw/DE_LU_2024/entsoe_joined_DE_LU.csv")
joined["timestamp_utc"] = pd.to_datetime(joined["timestamp_utc"], utc=True)
joined["date"] = joined["timestamp_utc"].dt.date

# Daily average price
daily_price = joined.groupby("date")["price_eur_mwh"].mean().reset_index()
daily_price.columns = ["t", "y_t"]

# Daily text: deduplicate outage reason texts across hours
def build_daily_text(group):
    texts = group["outage_texts"].dropna()
    texts = texts[texts != ""]
    if texts.empty:
        return ""
    unique = list(dict.fromkeys("; ".join(texts).split("; ")))
    unique = [t.strip() for t in unique if t.strip()]
    return ". ".join(unique[:10])

daily_text = joined.groupby("date").apply(build_daily_text).reset_index()
daily_text.columns = ["t", "text"]

result = daily_price.merge(daily_text, on="t")
result["t"] = result["t"].astype(str)
result = result.sort_values("t").reset_index(drop=True)

# Save as Parquet
import os
os.makedirs("data/entsoe", exist_ok=True)
result.to_parquet("data/entsoe/DE_LU_2024.parquet", index=False)
print(f"{len(result)} days, {(result.text != '').sum()} with text annotations")
```

### Step 4 — Visualize and sanity-check

**This step is not optional.** You need to answer: **do the REMIT annotations actually predict future price changes?**

#### 4a. Plot coverage

```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
dates = pd.to_datetime(result["t"])
has_text = result["text"] != ""

ax1.plot(dates, result["y_t"], lw=1.5, color="#2C3E50", label="Day-ahead price")
ax1.scatter(dates[has_text], result["y_t"][has_text], s=20, color="#E74C3C",
            zorder=5, label="Day with REMIT annotation")
ax1.set_ylabel("EUR/MWh")
ax1.legend(fontsize=8)
ax1.set_title("DE_LU Day-Ahead Prices 2024 — REMIT annotation coverage")

ax2.bar(dates, has_text.astype(int), color="#E74C3C", alpha=0.6)
ax2.set_ylabel("Annotated")
ax2.set_ylim(0, 1.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

plt.tight_layout()
plt.savefig("figures/entsoe_DE_LU_2024_coverage.png", dpi=150)
plt.show()
```

#### 4b. Check if outages predict price moves

```python
remit = pd.read_csv("entsoe_raw/DE_LU_2024/entsoe_remit_DE_LU.csv")
remit["event_start"] = pd.to_datetime(remit["event_start"])
remit["date"] = remit["event_start"].dt.date.astype(str)
unplanned_days = set(remit[remit["event_type"] == "A54"]["date"])

result["is_unplanned"] = result["t"].isin(unplanned_days)
result["price_change_3d"] = result["y_t"].shift(-3) - result["y_t"]

print("Avg 3-day price change after:")
print(f"  Unplanned outage : {result[result.is_unplanned]['price_change_3d'].mean():+.2f} EUR/MWh")
print(f"  Planned outage   : {result[has_text & ~result.is_unplanned]['price_change_3d'].mean():+.2f} EUR/MWh")
print(f"  No annotation    : {result[~has_text]['price_change_3d'].mean():+.2f} EUR/MWh")
```

#### 4c. Inspect the biggest events manually

Pick the top 5 unplanned outages by MW lost and plot price ±7 days around each. Did prices respond? Does the text actually describe something that would move the market?

#### 4d. Write up your findings

Produce a short doc at `docs/entsoe-annotation-findings.md` answering:

1. **Coverage** — what fraction of days have annotations?
2. **Predictive signal** — do unplanned outages show a measurable price response?
3. **Text quality** — are the reason texts informative or mostly noise?
4. **Verdict** — is this data worth scaling up?

If the annotations don't show predictive signal, say so clearly. A negative result here saves us from training on noise.

---

## Phase 2 — Build the Training Dataset

**Target: ~10,000 annotated daily rows to start.** That's roughly **5 zones × 5–6 years each** (~365 rows/year). More is better — aim for as much as is feasible.

### Zone and year selection

Prioritize **zone diversity** (different supply mixes = different text-price dynamics) and **years with strong signal**:

| Priority | Zone | Label | EIC Code | Why |
|----------|------|-------|----------|-----|
| 1 | Germany-Luxembourg | DE_LU | `10Y1001A1001A82H` | Largest market, richest REMIT |
| 2 | France | FR | `10YFR-RTE------C` | Nuclear-heavy, distinct dynamics |
| 3 | Spain | ES | `10YES-REE------0` | Solar-heavy, clean REMIT data |
| 4 | Poland | PL | `10YPL-AREA-----S` | Coal-heavy, different from above |
| 5 | Norway | NO_1 | `10YNO-1--------2` | Hydro-heavy |
| 6+ | Netherlands, Belgium, Austria, Finland | … | see below | If more data needed |

**Years to prioritize:** 2021–2022 first (extreme energy crisis = strongest text signal), then 2023–2024. Go back to 2019–2020 if more data is needed.

### Fetch each zone/year

```bash
# France 2022 (energy crisis — high signal)
ENTSOE_API_TOKEN=a447c244-ce1a-4568-bd84-cb601805204a uv run python scripts/fetch_entsoe.py \
  --start 202201010000 --end 202212310000 \
  --zone 10YFR-RTE------C --zone-label FR \
  --output-dir ./entsoe_raw/FR_2022

# Spain 2023
ENTSOE_API_TOKEN=a447c244-ce1a-4568-bd84-cb601805204a uv run python scripts/fetch_entsoe.py \
  --start 202301010000 --end 202312310000 \
  --zone 10YES-REE------0 --zone-label ES \
  --output-dir ./entsoe_raw/ES_2023
```

Each run takes ~3–4 hours. Run one at a time, or use separate machines for parallel runs on different zones.

For each run, repeat Steps 3–4 to produce a validated Parquet file at `data/entsoe/<ZONE>_<YEAR>.parquet`.

### Combine into training dataset

```python
import pandas as pd, glob

frames = []
for path in glob.glob("data/entsoe/*.parquet"):
    df = pd.read_parquet(path)
    # Derive series_id from filename, e.g. "DE_LU_2024"
    df["series_id"] = path.split("/")[-1].replace(".parquet", "")
    frames.append(df)

training = pd.concat(frames).reset_index(drop=True)
print(f"Total annotated rows : {len(training)}")
print(f"Annotated (text != ''): {(training.text != '').sum()}")
print(f"Series: {training.series_id.nunique()}")
print(training.groupby("series_id")[["y_t"]].count())

training.to_parquet("data/entsoe/entsoe_training.parquet", index=False)
```

---

## Acceptance Criteria

**Phase 1 — complete before scaling:**
- [ ] DE_LU 2024 fetched and joined (REMIT rows > 0)
- [ ] Daily Parquet at `data/entsoe/DE_LU_2024.parquet`
- [ ] Annotation coverage ≥ 50% of days
- [ ] Coverage plot saved to `figures/entsoe_DE_LU_2024_coverage.png`
- [ ] Event inspection plots for top 5 unplanned outages
- [ ] Written verdict at `docs/entsoe-annotation-findings.md`

**Phase 2 — training dataset:**
- [ ] At least 5 zones fetched and annotated
- [ ] At least 4 years per zone (prioritize 2021–2024)
- [ ] Zone diversity: nuclear, solar, coal, hydro each represented
- [ ] **Total annotated rows ≥ 10,000** (rows where `text != ""`)
- [ ] Combined training Parquet at `data/entsoe/entsoe_training.parquet`
- [ ] Row count and zone/year breakdown documented

**All committed to the `entsoe-bulk-download` branch.**

---

## Bidding Zone Reference

| Label | Zone | EIC Code |
|-------|------|----------|
| DE_LU | Germany-Luxembourg | `10Y1001A1001A82H` |
| FR | France | `10YFR-RTE------C` |
| ES | Spain | `10YES-REE------0` |
| NL | Netherlands | `10YNL----------L` |
| BE | Belgium | `10YBE----------2` |
| AT | Austria | `10YAT-APG------L` |
| PL | Poland | `10YPL-AREA-----S` |
| NO_1 | Norway (South) | `10YNO-1--------2` |
| FI | Finland | `10YFI-1--------U` |

---

## Questions / Context

Reach out to Raimi with questions. The relevant code is all on the `entsoe-bulk-download` branch. The model this data feeds into is documented in `notebooks/pyfiles/migas-1.5-inference-quickstart.py`.
