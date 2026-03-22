# ENTSO-E / Nord Pool Annotation Findings

**Zone:** NO1 (Norway South) — 2024
**Date:** 2026-03-20

---

## 1. Coverage

**92% of days have UMM annotations** (337 out of 366 days).

Gaps are concentrated in early January and a few scattered days. Coverage is well above the 50% threshold.

![Coverage plot](../figures/entsoe_NO1_2024_coverage.png)

---

## 2. Predictive Signal

3-day forward price change by annotation type:

| Condition | Avg price change (3d) |
|---|---|
| Unplanned outage | **+14.79 EUR/MWh** |
| No annotation | +2.85 EUR/MWh |
| Planned outage | -1.30 EUR/MWh |

**Unplanned outages show a strong, directionally correct signal** — 5x the no-annotation baseline. Planned outages are slightly negative, consistent with already-priced-in events. This is the pattern we'd expect if the text carries real information.

---

## 3. Text Quality

### Nordic zones (NO, FI, DK, SE) — good

Human-written descriptions with real operational detail:
- "Full refurbishment of Straumsmo G2 after fire"
- "Upgrade headrace tunnel"
- "Control system upgrade G1 + G2"
- "Weather conditions give icing on wind park"
- "repair of cooling pump"
- "Fault on the feed water pump"

NO1 has 42 unique reason texts across 119 events. FI has 193 unique reasons across 691 events.

### Non-Nordic zones (FR, BE) — poor

France has only 6 unique reason texts across 699 events: "Awaiting information", "Overhaul", "Test", "Start delay", "Technical failure", "Fuel Supply Failure". These are dropdown template values, not human descriptions.

### ENTSO-E API — no text at all

The ENTSO-E Transparency API returns structured outage data (asset name, fuel type code, capacity) but **no free-text reason descriptions**. The `<Reason>` XML element contains only a coded value (e.g. B19 = "Other"), never a `<text>` field. This is an API limitation, not a code bug.

---

## 4. Verdict

**Yes — the Nordic UMM data is worth scaling up.**

- Strong predictive signal from unplanned outage text
- High annotation coverage (92–100% across tested zones)
- Real human-written text with operational detail
- Free API, no authentication, fast to fetch (seconds per zone/year)

### Caveats

- **Limited to Nordic/Baltic zones** (NO, SE, FI, DK, EE, LT, LV, IE). Germany, France, Spain, Poland are not available through Nord Pool — they require EEX Transparency API (€550/month).
- **Zone diversity is narrower than originally planned.** Supply mix is mostly hydro + wind + CHP + nuclear (Nordic). No solar-heavy or coal-heavy zones available.
- **To reach 10,000 annotated rows:** 4 zones × 6 years (2019–2024) gives ~8,700 rows. Adding 1–2 more zones (NO2, SE3, DK2) pushes past the target.

### Recommendation

Proceed with Phase 2 using zones: **NO1, FI, DK1, SE1** as primary, adding **NO2, SE3** if more data is needed. Fetch ENTSO-E prices for 2019–2023 (existing bulk data only covers 2024), then pair with Nord Pool UMM text via `scripts/fetch_nordpool_umm.py`.
