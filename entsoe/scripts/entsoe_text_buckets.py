"""
Text bucket analysis — group similar reason texts, measure price impact per bucket.

Usage:
  uv run python scripts/entsoe_text_buckets.py
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)

DATA_DIR = os.path.join(_ROOT_DIR, "data/entsoe_examples_new")
FIG_DIR = os.path.join(_ROOT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────

umm_frames = []
for folder in sorted(glob.glob(f"{DATA_DIR}/*")):
    csv = glob.glob(f"{folder}/umm_parsed_*.csv")
    if not csv:
        continue
    df = pd.read_csv(csv[0])
    name = os.path.basename(folder)
    parts = name.rsplit("_", 1)
    df["zone"] = parts[0]
    df["year"] = int(parts[1])
    umm_frames.append(df)

umm = pd.concat(umm_frames).reset_index(drop=True)
umm["event_start"] = pd.to_datetime(umm["event_start"], format="ISO8601", utc=True)
umm["event_start_hour"] = umm["event_start"].dt.floor("h")

price_frames = []
for folder in sorted(glob.glob(f"{DATA_DIR}/*")):
    name = os.path.basename(folder)
    parts = name.rsplit("_", 1)
    zone, year = parts[0], int(parts[1])
    price_csv = glob.glob(f"{folder}/entsoe_prices_*.csv")
    if not price_csv:
        continue
    df = pd.read_csv(price_csv[0])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["zone"] = zone
    price_frames.append(df)

prices = pd.concat(price_frames).reset_index(drop=True)

print(f"UMM events: {len(umm):,}")
print(f"Hourly prices: {len(prices):,}")

# ── Define text buckets ──────────────────────────────────────────────────

def assign_bucket(text):
    if pd.isna(text):
        return "Unknown"
    t = str(text).lower().strip()

    # Specific patterns first (more specific → higher priority)
    if "icing" in t or ("ice " in t and "serv" not in t):
        return "Icing on wind turbines"
    if "weather" in t and ("wind" in t or "icing" in t or "reduced" in t):
        return "Icing on wind turbines"
    if "tripped" in t and "maintenance" in t and "icing" in t:
        return "Tripped + maintenance + icing"
    if "tripped" in t and "maintenance" in t:
        return "Tripped + maintenance"
    if "tripped" in t and "curtailment" in t:
        return "Tripped + curtailment"
    if "transformer" in t and ("trip" in t or "fault" in t or "limit" in t):
        return "Transformer fault"
    if "tripped" in t or "trip " in t:
        return "Tripped turbines"
    if "nuclear" in t or "ol3" in t or "olkiluoto" in t or "loviisa" in t or "scram" in t:
        return "Nuclear issue"
    if "system protection" in t or "fingrid" in t:
        return "Grid system protection"
    if "grid" in t and ("outage" in t or "maintenance" in t):
        return "Grid outage/maintenance"
    if "substation" in t:
        return "Substation work"
    if "coal mill" in t or "flue gas" in t:
        return "Coal plant failure"
    if "boiler" in t:
        return "Boiler issue"
    if "oil leak" in t or "oil spill" in t:
        return "Oil leak"
    if "vibration" in t:
        return "Vibration issue"
    if "communication" in t or "no communication" in t:
        return "Communication loss"
    if "cooling" in t:
        return "Cooling system issue"
    if "overload" in t or "overhaul" in t:
        return "Overhaul/Overload"
    if any(w in t for w in ["yearly maintenance", "annual maintenance", "foreseen maintenance"]):
        return "Scheduled annual maintenance"
    if "planned maintenance" in t or "planned reduction" in t:
        return "Planned maintenance"
    if "maintenance" in t or "maintenace" in t or "maitenance" in t:
        return "General maintenance"
    if "inspection" in t:
        return "Inspection"
    if "test" in t or "commission" in t:
        return "Testing/Commissioning"
    if any(w in t for w in ["repair", "repairment", "fix", "replace"]):
        return "Repair work"
    if any(w in t for w in ["failure", "fault", "faliure", "broken", "defect", "problem", "error", "malfunction"]):
        return "Technical failure"
    if "start" in t and ("fault" in t or "problem" in t or "not ready" in t or "block" in t):
        return "Start-up failure"
    if "unknown" in t:
        return "Unknown reason"
    if "new production" in t or "estimated max" in t:
        return "New unit commissioning"
    return "Other"


# ── Load LLM-classified buckets if available, else fall back to keywords ──

LLM_BUCKETS_PATH = os.path.join(_ROOT_DIR, "data/reason_text_buckets.json")
if os.path.exists(LLM_BUCKETS_PATH):
    import json
    with open(LLM_BUCKETS_PATH) as f:
        llm_mapping = json.load(f)
    print(f"Loaded LLM bucket mapping: {len(llm_mapping)} entries")
    umm["bucket"] = umm["reason_text"].map(llm_mapping).fillna("Other")
else:
    print("No LLM mapping found, using keyword buckets")
    umm["bucket"] = umm["reason_text"].apply(assign_bucket)

print(f"\nBuckets: {umm.bucket.nunique()}")
print("\nBucket distribution:")
for bucket, count in umm.bucket.value_counts().head(30).items():
    print(f"  {count:5d}x  {bucket}")

# ── Compute price impact per bucket ──────────────────────────────────────

HORIZONS_COARSE = [1, 3, 6, 12, 24]  # for heatmaps/boxplots
HORIZONS_FINE = list(range(1, 25))    # every hour for trajectories

# ── Build hourly seasonal profile per zone ────────────────────────────────
# Average price by hour-of-day, used to deseasonalize

print("Building hourly seasonal profiles per zone...")
seasonal_profiles = {}
for zone in prices.zone.unique():
    zp = prices[prices.zone == zone].copy()
    zp["hour"] = zp.timestamp_utc.dt.hour
    profile = zp.groupby("hour")["price_eur_mwh"].mean()
    seasonal_profiles[zone] = profile
    print(f"  {zone}: min={profile.min():.1f} (h={profile.idxmin()}), max={profile.max():.1f} (h={profile.idxmax()})")

results = []
for zone in umm.zone.unique():
    zone_prices = prices[prices.zone == zone].set_index("timestamp_utc")["price_eur_mwh"].sort_index()
    zone_umm = umm[umm.zone == zone]
    profile = seasonal_profiles.get(zone)

    for _, evt in zone_umm.iterrows():
        t0 = evt["event_start_hour"]
        if t0 not in zone_prices.index:
            continue
        p0 = zone_prices.loc[t0]
        h0 = t0.hour  # hour-of-day at event start

        row = {
            "bucket": evt["bucket"],
            "unavailability_type": evt["unavailability_type"],
            "zone": zone,
            "unavailable_mw": evt.get("unavailable_mw", 0),
        }
        for h in HORIZONS_FINE:
            th = t0 + pd.Timedelta(hours=h)
            if th in zone_prices.index:
                raw_change = zone_prices.loc[th] - p0
                # Expected change from daily cycle alone
                seasonal_change = profile.loc[th.hour] - profile.loc[h0]
                row[f"change_{h}h"] = raw_change - seasonal_change
            else:
                row[f"change_{h}h"] = np.nan
        results.append(row)

impacts = pd.DataFrame(results)
print(f"\nEvents with price match: {len(impacts):,}")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 1: Heatmap — bucket × horizon, showing mean price change
# ══════════════════════════════════════════════════════════════════════════

# Filter to buckets with enough data
bucket_counts = impacts.bucket.value_counts()
good_buckets = bucket_counts[bucket_counts >= 50].index.tolist()
filtered = impacts[impacts.bucket.isin(good_buckets)]

# Build matrix
matrix = []
for bucket in good_buckets:
    b_data = filtered[filtered.bucket == bucket]
    row = {"bucket": bucket, "n": len(b_data)}
    for h in HORIZONS_COARSE:
        row[f"+{h}h"] = b_data[f"change_{h}h"].mean()
    matrix.append(row)

mdf = pd.DataFrame(matrix).sort_values("+12h")

fig, ax = plt.subplots(figsize=(12, max(8, len(mdf) * 0.45)))

horizon_cols = [f"+{h}h" for h in HORIZONS_COARSE]
data = mdf[horizon_cols].values

vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
im = ax.imshow(data, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

ax.set_xticks(range(len(HORIZONS_COARSE)))
ax.set_xticklabels([f"+{h}h" for h in HORIZONS_COARSE], fontsize=11)
ax.set_yticks(range(len(mdf)))
labels = [f"{row.bucket} (n={row.n:,})" for _, row in mdf.iterrows()]
ax.set_yticklabels(labels, fontsize=9)

for i in range(len(mdf)):
    for j in range(len(HORIZONS_COARSE)):
        val = data[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(val) > vmax * 0.6 else "black")

plt.colorbar(im, ax=ax, label="Avg price change (EUR/MWh)", shrink=0.8)
ax.set_title("Price Impact by Text Bucket × Time Horizon\n(single-point, all zones 2016–2025)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Hours after outage start")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_bucket_heatmap.png", dpi=150, bbox_inches="tight")
print(f"Saved {FIG_DIR}/entsoe_bucket_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 2: Trajectory lines per bucket
# ══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 7))

# Build fine-grained means for all buckets
fine_matrix = []
for bucket in good_buckets:
    b_data = filtered[filtered.bucket == bucket]
    row = {"bucket": bucket, "n": len(b_data)}
    for h in HORIZONS_FINE:
        row[f"+{h}h"] = b_data[f"change_{h}h"].mean()
    vals = [abs(row[f"+{h}h"]) for h in HORIZONS_FINE if not np.isnan(row[f"+{h}h"])]
    row["max_abs"] = max(vals) if vals else 0
    fine_matrix.append(row)

fine_df = pd.DataFrame(fine_matrix)
top_buckets = fine_df.nlargest(12, "max_abs")

colors = plt.cm.tab20(np.linspace(0, 1, len(top_buckets)))

for i, (_, row) in enumerate(top_buckets.iterrows()):
    means = [row[f"+{h}h"] for h in HORIZONS_FINE]
    ax.plot(HORIZONS_FINE, means, linewidth=2, color=colors[i],
            label=f"{row.bucket} (n={row.n:,})")

ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_xlabel("Hours after outage start", fontsize=12)
ax.set_ylabel("Average price change (EUR/MWh)", fontsize=12)
ax.set_xticks(HORIZONS_FINE)
ax.set_xticklabels([f"{h}" for h in HORIZONS_FINE], fontsize=8)
ax.set_title("Hourly Price Trajectory by Text Bucket\n(top 12 buckets, all zones 2016–2025)",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=7, loc="best", ncol=2)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_bucket_trajectories.png", dpi=150)
print(f"Saved {FIG_DIR}/entsoe_bucket_trajectories.png")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 3: Boxplots for top buckets at +6h
# ══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 7))

# Top 15 buckets by count
top15 = bucket_counts.head(15).index.tolist()
box_data = []
box_labels = []

for bucket in top15:
    vals = filtered.loc[filtered.bucket == bucket, "change_6h"].dropna()
    box_data.append(vals)
    n = len(vals)
    mean = vals.mean()
    box_labels.append(f"{bucket}\n(n={n:,}, μ={mean:+.1f})")

bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=1.5), vert=True)

# Color by mean direction
for i, patch in enumerate(bp["boxes"]):
    mean = box_data[i].mean()
    patch.set_facecolor("#E74C3C" if mean > 0 else "#3498DB")
    patch.set_alpha(0.6)

# Mark means
for i, d in enumerate(box_data):
    ax.plot(i + 1, d.mean(), "D", color="black", markersize=6, zorder=5)

ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_ylabel("Price change at +6h (EUR/MWh)")
ax.set_title("Price Change Distribution at +6h by Text Bucket\n(top 15 buckets, all zones 2016–2025)",
             fontsize=14, fontweight="bold")
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_bucket_boxplot_6h.png", dpi=150, bbox_inches="tight")
print(f"Saved {FIG_DIR}/entsoe_bucket_boxplot_6h.png")

# ══════════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("TEXT BUCKET SUMMARY")
print("=" * 80)
print(f"{'Bucket':<35s} {'n':>6s} {'%unpl':>6s} {'+1h':>7s} {'+3h':>7s} {'+6h':>7s} {'+12h':>7s} {'+24h':>7s}")
print("-" * 80)

for _, row in mdf.iterrows():
    b = impacts[impacts.bucket == row.bucket]
    pct_unplanned = 100 * (b.unavailability_type == "Unplanned").mean()
    print(f"{row.bucket:<35s} {row.n:>6.0f} {pct_unplanned:>5.0f}% "
          f"{row['+1h']:>+6.1f} {row['+3h']:>+6.1f} {row['+6h']:>+6.1f} "
          f"{row['+12h']:>+6.1f} {row['+24h']:>+6.1f}")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 4: Per-zone heatmaps
# ══════════════════════════════════════════════════════════════════════════

zones = sorted(impacts.zone.unique())

for zone in zones:
    zone_impacts = impacts[impacts.zone == zone]

    # Buckets with enough data in this zone
    zone_bucket_counts = zone_impacts.bucket.value_counts()
    zone_good = zone_bucket_counts[zone_bucket_counts >= 20].index.tolist()
    zone_filtered = zone_impacts[zone_impacts.bucket.isin(zone_good)]

    if len(zone_good) < 3:
        continue

    zone_matrix = []
    for bucket in zone_good:
        b_data = zone_filtered[zone_filtered.bucket == bucket]
        row = {"bucket": bucket, "n": len(b_data)}
        for h in HORIZONS_COARSE:
            row[f"+{h}h"] = b_data[f"change_{h}h"].mean()
        zone_matrix.append(row)

    zmdf = pd.DataFrame(zone_matrix).sort_values("+12h")

    fig, ax = plt.subplots(figsize=(12, max(5, len(zmdf) * 0.45)))

    data = zmdf[horizon_cols].values
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 1)
    im = ax.imshow(data, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(HORIZONS_COARSE)))
    ax.set_xticklabels([f"+{h}h" for h in HORIZONS_COARSE], fontsize=11)
    ax.set_yticks(range(len(zmdf)))
    labels = [f"{row.bucket} (n={row.n:,})" for _, row in zmdf.iterrows()]
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(zmdf)):
        for j in range(len(HORIZONS_COARSE)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(val) > vmax * 0.6 else "black")

    plt.colorbar(im, ax=ax, label="Avg price change (EUR/MWh)", shrink=0.8)
    ax.set_title(f"{zone} — Price Impact by Text Bucket × Time Horizon\n(2016–2025)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Hours after outage start")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/entsoe_bucket_heatmap_{zone}.png", dpi=150, bbox_inches="tight")
    print(f"Saved {FIG_DIR}/entsoe_bucket_heatmap_{zone}.png")
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
# PLOT 5: Per-zone trajectory lines
# ══════════════════════════════════════════════════════════════════════════

for zone in zones:
    zone_impacts = impacts[impacts.zone == zone]
    zone_bucket_counts = zone_impacts.bucket.value_counts()
    zone_good = zone_bucket_counts[zone_bucket_counts >= 20].index.tolist()
    zone_filtered = zone_impacts[zone_impacts.bucket.isin(zone_good)]

    if len(zone_good) < 3:
        continue

    # Build fine-grained means per bucket
    zone_means = []
    for bucket in zone_good:
        b_data = zone_filtered[zone_filtered.bucket == bucket]
        row = {"bucket": bucket, "n": len(b_data)}
        for h in HORIZONS_FINE:
            row[f"+{h}h"] = b_data[f"change_{h}h"].mean()
        vals = [abs(row[f"+{h}h"]) for h in HORIZONS_FINE if not np.isnan(row[f"+{h}h"])]
        row["max_abs"] = max(vals) if vals else 0
        zone_means.append(row)

    zmdf = pd.DataFrame(zone_means).sort_values("max_abs", ascending=False)
    top = zmdf.head(10)

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, len(top)))

    for i, (_, row) in enumerate(top.iterrows()):
        means = [row[f"+{h}h"] for h in HORIZONS_FINE]
        ax.plot(HORIZONS_FINE, means, linewidth=2,
                color=colors[i], label=f"{row.bucket} (n={row.n:,})")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Hours after outage start", fontsize=12)
    ax.set_ylabel("Average price change (EUR/MWh)", fontsize=12)
    ax.set_xticks(HORIZONS_FINE)
    ax.set_xticklabels([f"{h}" for h in HORIZONS_FINE], fontsize=8)
    ax.set_title(f"{zone} — Hourly Price Trajectory by Text Bucket\n(top 10 buckets, 2016–2025)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/entsoe_bucket_trajectories_{zone}.png", dpi=150)
    print(f"Saved {FIG_DIR}/entsoe_bucket_trajectories_{zone}.png")
    plt.close()
