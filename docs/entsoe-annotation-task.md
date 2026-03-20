# Task: Build a Text-Annotated Energy Price Time Series

## Goal

Produce a set of well-annotated Parquet files — one per country/zone per year — that look like this:

| t | y_t | text |
|---|-----|------|
| 2024-01-01 | 52.3 | Planned: Ringhals 3 (Nuclear, 1063 MW unavailable). Upgrade headrace tunnel. Unplanned: Straumsmo (Reservoir hydro, 130 MW unavailable). Full refurbishment after fire. |
| 2024-01-02 | 48.1 | |
| … | … | … |

- **`t`** — date string `YYYY-MM-DD`
- **`y_t`** — daily average day-ahead electricity price (EUR/MWh)
- **`text`** — free-text summary of REMIT outage events active on that day

The target is **~10,000 annotated daily rows** to start. These files will be used to train Migas-1.5 on energy price forecasting with text conditioning.

**Use Parquet throughout, not CSV.**

---

## Key Finding: Data Sources

### ENTSO-E Transparency API — prices only, NO free text

The ENTSO-E API (document types A44, A77, A78) returns:
- Day-ahead prices (hourly) ✓
- REMIT outage events with asset name, fuel type code, capacity ✓
- **Free-text reason descriptions ✗** — the `<Reason>` XML element only contains a `<code>` (e.g. B19), never a `<text>` field

This is an API limitation, not a code bug. `scripts/fetch_entsoe.py` works correctly — it fetches structured outage data, but the text we need doesn't exist in this API.

### Nord Pool UMM API — real free text, Nordic zones only

The Nord Pool UMM API (`ummapi.nordpoolgroup.com`) provides:
- **Real human-written outage descriptions** in `unavailabilityReason` and `remarks` fields
- Examples: "Full refurbishment of Straumsmo G2 after fire", "Icing on blades", "repair of cooling pump"
- **Free, no API key** for read access
- Fast (seconds per zone/year vs hours for ENTSO-E)
- Historical data back to ~2013
- **Only Nordic/Baltic zones**: NO, SE, FI, DK, EE, LT, LV, IE

Text quality varies by country:
- **Good text** (42–193 unique reasons): NO, FI, DK, SE — real descriptions
- **Template text** (6 unique reasons): FR, BE — "Awaiting information", "Overhaul", etc.
- **Not available**: DE, ES, PL, AT, IT — these publish on EEX (€550/month paid API)

### Approach: Nord Pool UMM (text) + ENTSO-E (prices)

Use `scripts/fetch_nordpool_umm.py` for text annotations, ENTSO-E for day-ahead prices.

---

## What Has Already Been Done ✓

- **`scripts/fetch_nordpool_umm.py`** — fetches UMM messages with free text from Nord Pool, parses them, builds daily annotations, and joins with ENTSO-E prices. Supports `--zone`, `--year`, `--prices-csv`.
- **`scripts/fetch_entsoe.py`** — fetches day-ahead prices (A44) from ENTSO-E. Also fetches REMIT (A77/A78) but **without free text**.
- **`data/entsoe_bulk/`** — day-ahead prices for 12 European zones, 2024
- **`data/entsoe/`** — completed daily Parquet files:
  - `NO1_2024.parquet` — 366 days, 337 annotated (92%)
  - `FI_2024.parquet` — 366 days, 366 annotated (100%)
  - `DK1_2024.parquet` — 366 days, 340 annotated (93%)
  - `SE1_2024.parquet` — 366 days, 366 annotated (100%)

---

## Phase 1 — Validate One Zone First

**Already done for NO1 2024.** Text quality is confirmed good with real human descriptions.

### Step 1 — Setup

```bash
git clone git@github.com:Synthefy/synthefy-migas.git
cd synthefy-migas
git checkout ajan-entsoe
uv sync
```

### Step 2 — Fetch a zone

```bash
# Example: Norway 2024 (already done)
uv run python scripts/fetch_nordpool_umm.py \
  --zone NO1 --year 2024 \
  --prices-csv data/entsoe_bulk/day_ahead_prices/NO_1.csv
```

This produces (in seconds):
```
data/entsoe_raw/NO1_2024/umm_raw_NO1_2024.json     # raw API response
data/entsoe_raw/NO1_2024/umm_parsed_NO1_2024.csv    # parsed flat events
data/entsoe/NO1_2024.parquet                         # final daily (t, y_t, text)
```

### Step 3 — Visualize and sanity-check

**This step is not optional.** You need to answer: **do the REMIT annotations actually predict future price changes?**

#### 3a. Plot coverage

```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

result = pd.read_parquet("data/entsoe/NO1_2024.parquet")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
dates = pd.to_datetime(result["t"])
has_text = result["text"] != ""

ax1.plot(dates, result["y_t"], lw=1.5, color="#2C3E50", label="Day-ahead price")
ax1.scatter(dates[has_text], result["y_t"][has_text], s=20, color="#E74C3C",
            zorder=5, label="Day with UMM annotation")
ax1.set_ylabel("EUR/MWh")
ax1.legend(fontsize=8)
ax1.set_title("NO1 Day-Ahead Prices 2024 — UMM annotation coverage")

ax2.bar(dates, has_text.astype(int), color="#E74C3C", alpha=0.6)
ax2.set_ylabel("Annotated")
ax2.set_ylim(0, 1.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

plt.tight_layout()
plt.savefig("figures/entsoe_NO1_2024_coverage.png", dpi=150)
plt.show()
```

#### 3b. Check if outages predict price moves

```python
umm = pd.read_csv("data/entsoe_raw/NO1_2024/umm_parsed_NO1_2024.csv")
umm["date"] = pd.to_datetime(umm["event_start"]).dt.strftime("%Y-%m-%d")
unplanned_days = set(umm[umm["unavailability_type"] == "Unplanned"]["date"])

result["is_unplanned"] = result["t"].isin(unplanned_days)
result["price_change_3d"] = result["y_t"].shift(-3) - result["y_t"]

print("Avg 3-day price change after:")
print(f"  Unplanned outage : {result[result.is_unplanned]['price_change_3d'].mean():+.2f} EUR/MWh")
print(f"  Planned outage   : {result[has_text & ~result.is_unplanned]['price_change_3d'].mean():+.2f} EUR/MWh")
print(f"  No annotation    : {result[~has_text]['price_change_3d'].mean():+.2f} EUR/MWh")
```

#### 3c. Write up findings

Produce `docs/entsoe-annotation-findings.md` answering:

1. **Coverage** — what fraction of days have annotations?
2. **Predictive signal** — do unplanned outages show a measurable price response?
3. **Text quality** — are the reason texts informative or mostly noise?
4. **Verdict** — is this data worth scaling up?

---

## Phase 2 — Build the Training Dataset

**Target: ~10,000 annotated daily rows.**

### Zone and year selection

| Priority | Zone | Label | Supply mix | Text quality |
|----------|------|-------|------------|--------------|
| 1 | Norway (South) | NO1 | Hydro-heavy | Good (42 unique reasons) |
| 2 | Finland | FI | Nuclear + CHP | Good (193 unique reasons) |
| 3 | Denmark (West) | DK1 | Wind + gas | Good (144 unique reasons) |
| 4 | Sweden (North) | SE1 | Hydro + wind | Good (102 unique reasons) |
| 5 | Norway (South-West) | NO2 | Hydro | Good (likely) |
| 6 | Sweden (South) | SE3 | Mixed | Good (likely) |
| 7 | Denmark (East) | DK2 | Wind + thermal | Good (likely) |

**Years:** 2019–2024 (6 years × 4+ zones ≈ 8,700+ days). Nord Pool data goes back to ~2013.

**Note:** ENTSO-E bulk prices in `data/entsoe_bulk/` only cover **2024**. For earlier years, fetch prices first via `scripts/fetch_entsoe.py` (prices only — fast, minutes per zone/year).

### Fetch prices for earlier years

```bash
# Example: Norway 2022 prices
ENTSOE_API_TOKEN=a447c244-ce1a-4568-bd84-cb601805204a uv run python scripts/fetch_entsoe.py \
  --start 202201010000 --end 202212310000 \
  --zone 10YNO-1--------2 --zone-label NO1 \
  --output-dir ./data/entsoe_raw/NO1_2022
```

> Only the price fetch matters here (~minutes). The REMIT fetch will run but produce no text — that's expected.

### Fetch UMMs for each zone/year

```bash
uv run python scripts/fetch_nordpool_umm.py \
  --zone NO1 --year 2022 \
  --prices-csv data/entsoe_raw/NO1_2022/entsoe_prices_NO1.csv
```

### Batch: all zones × all years

```bash
ZONES="NO1 FI DK1 SE1"
YEARS="2019 2020 2021 2022 2023 2024"

for zone in $ZONES; do
  for year in $YEARS; do
    echo "=== $zone $year ==="
    uv run python scripts/fetch_nordpool_umm.py \
      --zone $zone --year $year \
      --prices-csv data/entsoe_raw/${zone}_${year}/entsoe_prices_${zone}.csv
  done
done
```

> **Prices must be fetched first** for years without existing data. 2024 prices exist in `data/entsoe_bulk/`. For 2019–2023, run `fetch_entsoe.py` first.

### Combine into training dataset

```python
import pandas as pd, glob

frames = []
for path in sorted(glob.glob("data/entsoe/*.parquet")):
    if "training" in path:
        continue
    df = pd.read_parquet(path)
    df["series_id"] = path.split("/")[-1].replace(".parquet", "")
    frames.append(df)

training = pd.concat(frames).reset_index(drop=True)
print(f"Total rows:           {len(training)}")
print(f"Annotated (text!=''): {(training.text != '').sum()}")
print(f"Series:               {training.series_id.nunique()}")
print(training.groupby("series_id")[["y_t"]].count())

training.to_parquet("data/entsoe/entsoe_training.parquet", index=False)
```

---

## Acceptance Criteria

**Phase 1 — complete before scaling:**
- [x] NO1 2024 fetched with real UMM text
- [x] Daily Parquet at `data/entsoe/NO1_2024.parquet`
- [x] Annotation coverage ≥ 50% of days (92%)
- [ ] Coverage plot saved to `figures/entsoe_NO1_2024_coverage.png`
- [ ] Event inspection plots for top 5 unplanned outages
- [ ] Written verdict at `docs/entsoe-annotation-findings.md`

**Phase 2 — training dataset:**
- [ ] At least 4 zones fetched and annotated
- [ ] At least 4 years per zone (prioritize 2021–2024)
- [ ] Zone diversity: hydro, nuclear, wind, CHP represented
- [ ] **Total annotated rows ≥ 10,000** (rows where `text != ""`)
- [ ] Combined training Parquet at `data/entsoe/entsoe_training.parquet`
- [ ] Row count and zone/year breakdown documented

**All committed to the `ajan-entsoe` branch.**

---

## Available Zone Reference

### Nord Pool UMM zones (have free text)

| Label | Zone | EIC Code | Supply mix |
|-------|------|----------|------------|
| NO1 | Norway (South) | `10YNO-1--------2` | Hydro |
| NO2 | Norway (South-West) | `10YNO-2--------T` | Hydro |
| NO4 | Norway (North) | `10YNO-4--------9` | Hydro |
| NO5 | Norway (West) | `10Y1001A1001A48H` | Hydro |
| SE1 | Sweden (North) | `10Y1001A1001A44P` | Hydro + wind |
| SE2 | Sweden (North-Central) | `10Y1001A1001A45N` | Hydro + wind |
| SE3 | Sweden (South-Central) | `10Y1001A1001A46L` | Nuclear + mixed |
| SE4 | Sweden (South) | `10Y1001A1001A47J` | Mixed |
| FI | Finland | `10YFI-1--------U` | Nuclear + CHP |
| DK1 | Denmark (West) | `10YDK-1--------W` | Wind + gas |
| DK2 | Denmark (East) | `10YDK-2--------M` | Wind + thermal |
| EE | Estonia | `10Y1001A1001A39I` | Oil shale |
| LT | Lithuania | `10YLT-1001A0008Q` | Mixed |
| LV | Latvia | `10YLV-1001A00074` | Hydro |
| IE | Ireland | `10Y1001A1001A59C` | Gas + wind |

### Zones with prices but no free text (need EEX, €550/month)

| Label | Zone | EIC Code |
|-------|------|----------|
| DE_LU | Germany-Luxembourg | `10Y1001A1001A82H` |
| FR | France | `10YFR-RTE------C` |
| ES | Spain | `10YES-REE------0` |
| PL | Poland | `10YPL-AREA-----S` |
| AT | Austria | `10YAT-APG------L` |
| IT_N | Italy (North) | `10Y1001A1001A73I` |

---

## Questions / Context

Reach out to Raimi with questions. The relevant code is all on the `ajan-entsoe` branch. The model this data feeds into is documented in `notebooks/pyfiles/migas-1.5-inference-quickstart.py`.
