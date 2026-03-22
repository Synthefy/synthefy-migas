"""
Deep analysis of LLM-classified outage buckets.

Analyses:
  1. Bucket x MW severity heatmap — does size matter differently per bucket?
  2. Bucket x zone heatmap — which buckets hit which zones hardest?
  3. Recovery time by bucket — how long until price returns to baseline?
  4. Cascade detection — do simultaneous outages cause superlinear price impact?
  5. Seasonal patterns — which buckets cluster in winter vs summer?
  6. Training data signal audit — which buckets are worth including in text?

All deseasonalized (subtract hourly profile per zone).

Usage:
  uv run python scripts/entsoe_bucket_deep_analysis.py
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)

DATA_DIR = os.path.join(_ROOT_DIR, "data/entsoe_examples_new")
FIG_DIR = os.path.join(_ROOT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

BUCKETS_PATH = os.path.join(_ROOT_DIR, "data/reason_text_buckets.json")

# ── Load data ─────────────────────────────────────────────────────────────

umm_frames = []
for folder in sorted(glob.glob(f"{DATA_DIR}/*")):
    csv_files = glob.glob(f"{folder}/umm_parsed_*.csv")
    if not csv_files:
        continue
    df = pd.read_csv(csv_files[0])
    name = os.path.basename(folder)
    parts = name.rsplit("_", 1)
    df["zone"] = parts[0]
    df["year"] = int(parts[1])
    umm_frames.append(df)

umm = pd.concat(umm_frames).reset_index(drop=True)
umm["event_start"] = pd.to_datetime(umm["event_start"], format="ISO8601", utc=True)
umm["event_end"] = pd.to_datetime(umm["event_end"], format="ISO8601", utc=True)
umm["event_start_hour"] = umm["event_start"].dt.floor("h")

price_frames = []
for folder in sorted(glob.glob(f"{DATA_DIR}/*")):
    name = os.path.basename(folder)
    parts = name.rsplit("_", 1)
    zone = parts[0]
    price_csv = glob.glob(f"{folder}/entsoe_prices_*.csv")
    if not price_csv:
        continue
    df = pd.read_csv(price_csv[0])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["zone"] = zone
    price_frames.append(df)

prices = pd.concat(price_frames).reset_index(drop=True)

# Load LLM buckets
with open(BUCKETS_PATH) as f:
    llm_mapping = json.load(f)
umm["bucket"] = umm["reason_text"].map(llm_mapping).fillna("Other")

print(f"UMM events: {len(umm):,}")
print(f"Hourly prices: {len(prices):,}")
print(f"LLM bucket mapping: {len(llm_mapping)} entries")

# ── Seasonal profiles ─────────────────────────────────────────────────────

seasonal_profiles = {}
for zone in prices.zone.unique():
    zp = prices[prices.zone == zone].copy()
    zp["hour"] = zp.timestamp_utc.dt.hour
    seasonal_profiles[zone] = zp.groupby("hour")["price_eur_mwh"].mean()

# ── Compute deseasonalized price changes at fine resolution ───────────────

HORIZONS = list(range(1, 25))

results = []
for zone in umm.zone.unique():
    zone_prices = prices[prices.zone == zone].set_index("timestamp_utc")["price_eur_mwh"].sort_index()
    zone_umm = umm[umm.zone == zone]
    profile = seasonal_profiles.get(zone)
    if profile is None:
        continue

    for _, evt in zone_umm.iterrows():
        t0 = evt["event_start_hour"]
        if t0 not in zone_prices.index:
            continue
        p0 = zone_prices.loc[t0]
        h0 = t0.hour

        row = {
            "bucket": evt["bucket"],
            "unavailability_type": evt["unavailability_type"],
            "zone": zone,
            "unavailable_mw": evt.get("unavailable_mw", 0),
            "fuel_type": evt.get("fuel_type", ""),
            "event_start": t0,
            "month": t0.month,
        }
        for h in HORIZONS:
            th = t0 + pd.Timedelta(hours=h)
            if th in zone_prices.index:
                raw = zone_prices.loc[th] - p0
                seasonal = profile.loc[th.hour] - profile.loc[h0]
                row[f"change_{h}h"] = raw - seasonal
            else:
                row[f"change_{h}h"] = np.nan
        results.append(row)

impacts = pd.DataFrame(results)
print(f"Events with price match: {len(impacts):,}")

# Filter out tiny buckets
bucket_counts = impacts.bucket.value_counts()
good_buckets = bucket_counts[bucket_counts >= 30].index.tolist()
filtered = impacts[impacts.bucket.isin(good_buckets)]

print(f"Buckets with 30+ events: {len(good_buckets)}")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 1: Bucket × MW Severity Heatmap
# ══════════════════════════════════════════════════════════════════════════

print("\n--- Plot 1: Bucket x MW severity ---")

# Create MW quartile bins
mw_valid = filtered["unavailable_mw"].dropna()
mw_valid = mw_valid[mw_valid > 0]
q25, q50, q75 = mw_valid.quantile([0.25, 0.5, 0.75])

mw_bins = [
    (0, q25, f"< {q25:.0f} MW"),
    (q25, q50, f"{q25:.0f}–{q50:.0f} MW"),
    (q50, q75, f"{q50:.0f}–{q75:.0f} MW"),
    (q75, 999999, f"> {q75:.0f} MW"),
]

# Horizons for this heatmap
H_SHOW = [1, 3, 6, 12, 24]

# Sort buckets by overall +6h impact
bucket_order = (
    filtered.groupby("bucket")["change_6h"]
    .mean()
    .sort_values()
    .index.tolist()
)
bucket_order = [b for b in bucket_order if b in good_buckets]

fig, axes = plt.subplots(1, len(mw_bins), figsize=(24, max(8, len(bucket_order) * 0.4)),
                         sharey=True)

all_vals = []
for lo, hi, label in mw_bins:
    for bucket in bucket_order:
        sub = filtered[(filtered.bucket == bucket) &
                       (filtered.unavailable_mw >= lo) &
                       (filtered.unavailable_mw < hi)]
        for h in H_SHOW:
            v = sub[f"change_{h}h"].mean()
            if not np.isnan(v):
                all_vals.append(v)

vmax = max(abs(v) for v in all_vals) if all_vals else 10

for ax_i, (lo, hi, mw_label) in enumerate(mw_bins):
    matrix = []
    ns = []
    for bucket in bucket_order:
        sub = filtered[(filtered.bucket == bucket) &
                       (filtered.unavailable_mw >= lo) &
                       (filtered.unavailable_mw < hi)]
        row_vals = []
        for h in H_SHOW:
            row_vals.append(sub[f"change_{h}h"].mean() if len(sub) >= 5 else np.nan)
        matrix.append(row_vals)
        ns.append(len(sub))

    data = np.array(matrix)
    ax = axes[ax_i]
    im = ax.imshow(data, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(H_SHOW)))
    ax.set_xticklabels([f"+{h}h" for h in H_SHOW], fontsize=9)
    ax.set_title(f"{mw_label}", fontsize=12, fontweight="bold")

    if ax_i == 0:
        ax.set_yticks(range(len(bucket_order)))
        labels = [f"{b} (n={ns[i]})" for i, b in enumerate(bucket_order)]
        ax.set_yticklabels(labels, fontsize=8)
    else:
        # Show just n counts
        ax.set_yticks(range(len(bucket_order)))
        ax.set_yticklabels([f"n={ns[i]}" for i in range(len(bucket_order))], fontsize=7)

    for i in range(len(bucket_order)):
        for j in range(len(H_SHOW)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(val) > vmax * 0.6 else "black")

plt.colorbar(im, ax=axes, label="Avg deseasonalized price change (EUR/MWh)", shrink=0.6)
fig.suptitle("Price Impact: Bucket × MW Severity\n(deseasonalized, all zones 2016–2025)",
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_bucket_x_mw_heatmap.png", dpi=150, bbox_inches="tight")
print(f"Saved {FIG_DIR}/entsoe_bucket_x_mw_heatmap.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# PLOT 2: Bucket × Zone Heatmap at +6h
# ══════════════════════════════════════════════════════════════════════════

print("\n--- Plot 2: Bucket x zone ---")

zones = sorted(filtered.zone.unique())

# Use +6h as the main horizon
matrix = []
row_labels = []
for bucket in bucket_order:
    row_vals = []
    for zone in zones:
        sub = filtered[(filtered.bucket == bucket) & (filtered.zone == zone)]
        row_vals.append(sub["change_6h"].mean() if len(sub) >= 10 else np.nan)
    matrix.append(row_vals)
    n_total = len(filtered[filtered.bucket == bucket])
    row_labels.append(f"{bucket} (n={n_total})")

data = np.array(matrix)
vmax_z = np.nanmax(np.abs(data[np.isfinite(data)])) if np.any(np.isfinite(data)) else 10

fig, ax = plt.subplots(figsize=(14, max(8, len(bucket_order) * 0.4)))
im = ax.imshow(data, cmap="RdBu_r", aspect="auto", vmin=-vmax_z, vmax=vmax_z)

ax.set_xticks(range(len(zones)))
ax.set_xticklabels(zones, fontsize=11, fontweight="bold")
ax.set_yticks(range(len(bucket_order)))
ax.set_yticklabels(row_labels, fontsize=8)

for i in range(len(bucket_order)):
    for j in range(len(zones)):
        val = data[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(val) > vmax_z * 0.6 else "black")

plt.colorbar(im, ax=ax, label="Avg price change at +6h (EUR/MWh)", shrink=0.8)
ax.set_title("Price Impact at +6h: Bucket × Zone\n(deseasonalized, 2016–2025)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_bucket_x_zone_heatmap.png", dpi=150, bbox_inches="tight")
print(f"Saved {FIG_DIR}/entsoe_bucket_x_zone_heatmap.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# PLOT 3: Recovery time by bucket — hours until mean change crosses zero
# ══════════════════════════════════════════════════════════════════════════

print("\n--- Plot 3: Recovery time ---")

recovery = []
for bucket in good_buckets:
    b_data = filtered[filtered.bucket == bucket]
    trajectory = [b_data[f"change_{h}h"].mean() for h in HORIZONS]

    # Find first crossing from positive to <=0 (or stays positive)
    peak_h = 0
    peak_val = 0
    for i, (h, v) in enumerate(zip(HORIZONS, trajectory)):
        if not np.isnan(v) and v > peak_val:
            peak_val = v
            peak_h = h

    # Recovery: first hour after peak where value <= 0
    recovery_h = 24  # default: didn't recover
    past_peak = False
    for h, v in zip(HORIZONS, trajectory):
        if h >= peak_h:
            past_peak = True
        if past_peak and not np.isnan(v) and v <= 0:
            recovery_h = h
            break

    recovery.append({
        "bucket": bucket,
        "n": len(b_data),
        "peak_change": peak_val,
        "peak_hour": peak_h,
        "recovery_hour": recovery_h,
        "change_24h": trajectory[-1] if not np.isnan(trajectory[-1]) else 0,
    })

rdf = pd.DataFrame(recovery).sort_values("peak_change", ascending=False)

# Only show buckets with positive peak (price increase)
rdf_pos = rdf[rdf.peak_change > 0.5]

fig, ax = plt.subplots(figsize=(14, max(6, len(rdf_pos) * 0.45)))

colors = []
for _, row in rdf_pos.iterrows():
    if row.recovery_hour <= 6:
        colors.append("#2ECC71")  # fast recovery
    elif row.recovery_hour <= 12:
        colors.append("#F39C12")  # medium
    else:
        colors.append("#E74C3C")  # slow/no recovery

y_pos = range(len(rdf_pos))
bars = ax.barh(y_pos, rdf_pos.peak_change, color=colors, alpha=0.7, edgecolor="white")

# Annotate with peak hour and recovery hour
for i, (_, row) in enumerate(rdf_pos.iterrows()):
    rec_text = f"{row.recovery_hour}h" if row.recovery_hour < 24 else "24h+"
    ax.text(row.peak_change + 0.3, i,
            f"peak +{row.peak_hour}h, recover {rec_text} (n={row.n})",
            va="center", fontsize=8)

ax.set_yticks(y_pos)
ax.set_yticklabels(rdf_pos.bucket, fontsize=9)
ax.set_xlabel("Peak deseasonalized price change (EUR/MWh)", fontsize=11)
ax.legend(handles=[
    mpatches.Patch(color="#2ECC71", alpha=0.7, label="Recovery <= 6h"),
    mpatches.Patch(color="#F39C12", alpha=0.7, label="Recovery 7-12h"),
    mpatches.Patch(color="#E74C3C", alpha=0.7, label="Recovery > 12h / none"),
], loc="lower right", fontsize=9)
ax.set_title("Peak Price Impact & Recovery Time by Bucket\n(deseasonalized, all zones 2016–2025)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_bucket_recovery.png", dpi=150, bbox_inches="tight")
print(f"Saved {FIG_DIR}/entsoe_bucket_recovery.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# PLOT 4: Cascade detection — price impact when N outages overlap
# ══════════════════════════════════════════════════════════════════════════

print("\n--- Plot 4: Cascade detection ---")

# For each event, count how many OTHER events are active at the same time in the same zone
# Active = event_start <= t0 < event_end
umm_active = umm.dropna(subset=["event_start", "event_end"]).copy()

# Build a count of concurrent events per zone at each event start
concurrent_counts = []
for zone in umm_active.zone.unique():
    zone_evts = umm_active[umm_active.zone == zone].sort_values("event_start")
    starts = zone_evts["event_start"].values
    ends = zone_evts["event_end"].values

    for i in range(len(zone_evts)):
        t = starts[i]
        # Count events where start <= t < end (excluding self)
        active = np.sum((starts[:i] <= t) & (ends[:i] > t)) + \
                 np.sum((starts[i+1:] <= t) & (ends[i+1:] > t))
        concurrent_counts.append(active)

umm_active = umm_active.copy()
umm_active["concurrent"] = concurrent_counts

# Merge concurrent count into impacts
impacts_cascade = impacts.merge(
    umm_active[["event_start", "zone", "concurrent"]],
    on=["event_start", "zone"],
    how="left"
)
impacts_cascade["concurrent"] = impacts_cascade["concurrent"].fillna(0).astype(int)

# Bin: 0 (solo), 1-2, 3-5, 6+
cascade_bins = [
    (0, 0, "Solo (0 concurrent)"),
    (1, 2, "1–2 concurrent"),
    (3, 5, "3–5 concurrent"),
    (6, 999, "6+ concurrent"),
]

fig, ax = plt.subplots(figsize=(12, 7))

cascade_colors = ["#95A5A6", "#3498DB", "#F39C12", "#E74C3C"]

for (lo, hi, label), color in zip(cascade_bins, cascade_colors):
    sub = impacts_cascade[
        (impacts_cascade.concurrent >= lo) & (impacts_cascade.concurrent <= hi)
    ]
    if len(sub) < 20:
        continue
    means = [sub[f"change_{h}h"].mean() for h in HORIZONS]
    ax.plot(HORIZONS, means, marker="o", linewidth=2.5, markersize=7,
            color=color, label=f"{label} (n={len(sub):,})")

ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_xlabel("Hours after outage start", fontsize=12)
ax.set_ylabel("Avg deseasonalized price change (EUR/MWh)", fontsize=12)
ax.set_xticks(HORIZONS)
ax.set_xticklabels([str(h) for h in HORIZONS], fontsize=8)
ax.set_title("Cascade Effect: Price Impact by Number of Concurrent Outages\n(all zones 2016–2025)",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_cascade_effect.png", dpi=150)
print(f"Saved {FIG_DIR}/entsoe_cascade_effect.png")
plt.close()

# ── Plot 4b: Cascade per zone ─────────────────────────────────────────────

print("\n--- Plot 4b: Cascade per zone ---")

cascade_zones = sorted(impacts_cascade.zone.unique())
n_zones = len(cascade_zones)
fig, axes = plt.subplots(n_zones, 1, figsize=(14, 4.5 * n_zones), sharex=True)

for ax_i, zone in enumerate(cascade_zones):
    ax = axes[ax_i]
    zone_data = impacts_cascade[impacts_cascade.zone == zone]

    for (lo, hi, label), color in zip(cascade_bins, cascade_colors):
        sub = zone_data[(zone_data.concurrent >= lo) & (zone_data.concurrent <= hi)]
        if len(sub) < 10:
            continue
        means = [sub[f"change_{h}h"].mean() for h in HORIZONS]
        ax.plot(HORIZONS, means, marker="o", linewidth=2, markersize=5,
                color=color, label=f"{label} (n={len(sub):,})")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("EUR/MWh", fontsize=10)
    ax.set_title(f"{zone}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="best", ncol=2)

axes[-1].set_xlabel("Hours after outage start", fontsize=12)
axes[-1].set_xticks(HORIZONS)
axes[-1].set_xticklabels([str(h) for h in HORIZONS], fontsize=8)

fig.suptitle("Cascade Effect by Zone\n(deseasonalized, 2016–2025)",
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_cascade_per_zone.png", dpi=150, bbox_inches="tight")
print(f"Saved {FIG_DIR}/entsoe_cascade_per_zone.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# PLOT 4c: Cascade by TOTAL MW offline (not event count)
# ══════════════════════════════════════════════════════════════════════════

print("\n--- Plot 4c: Cascade by total MW offline ---")

# For each event, compute total MW offline in that zone at event_start
umm_active_mw = umm.dropna(subset=["event_start", "event_end"]).copy()
umm_active_mw["unavailable_mw"] = umm_active_mw["unavailable_mw"].fillna(0)

total_mw_at_start = []
for zone in umm_active_mw.zone.unique():
    zone_evts = umm_active_mw[umm_active_mw.zone == zone].sort_values("event_start")
    starts = zone_evts["event_start"].values
    ends = zone_evts["event_end"].values
    mws = zone_evts["unavailable_mw"].values

    for i in range(len(zone_evts)):
        t = starts[i]
        # Sum MW of all active events at time t (including self)
        active_mask = (starts <= t) & (ends > t)
        total_mw = mws[active_mask].sum()
        total_mw_at_start.append(total_mw)

umm_active_mw = umm_active_mw.copy()
umm_active_mw["total_mw_offline"] = total_mw_at_start

# Merge into impacts
impacts_mw = impacts.merge(
    umm_active_mw[["event_start", "zone", "total_mw_offline"]],
    on=["event_start", "zone"],
    how="left"
)
impacts_mw["total_mw_offline"] = impacts_mw["total_mw_offline"].fillna(0)

# Bin by total MW offline
mw_q25, mw_q50, mw_q75 = impacts_mw["total_mw_offline"].quantile([0.25, 0.5, 0.75])

mw_cascade_bins = [
    (0, mw_q25, f"< {mw_q25:.0f} MW total"),
    (mw_q25, mw_q50, f"{mw_q25:.0f}–{mw_q50:.0f} MW"),
    (mw_q50, mw_q75, f"{mw_q50:.0f}–{mw_q75:.0f} MW"),
    (mw_q75, 999999, f"> {mw_q75:.0f} MW total"),
]

# All zones combined
fig, ax = plt.subplots(figsize=(12, 7))
mw_colors = ["#95A5A6", "#3498DB", "#F39C12", "#E74C3C"]

for (lo, hi, label), color in zip(mw_cascade_bins, mw_colors):
    sub = impacts_mw[(impacts_mw.total_mw_offline >= lo) & (impacts_mw.total_mw_offline < hi)]
    if len(sub) < 20:
        continue
    means = [sub[f"change_{h}h"].mean() for h in HORIZONS]
    ax.plot(HORIZONS, means, marker="o", linewidth=2.5, markersize=7,
            color=color, label=f"{label} (n={len(sub):,})")

ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_xlabel("Hours after outage start", fontsize=12)
ax.set_ylabel("Avg deseasonalized price change (EUR/MWh)", fontsize=12)
ax.set_xticks(HORIZONS)
ax.set_xticklabels([str(h) for h in HORIZONS], fontsize=8)
ax.set_title("Price Impact by Total MW Offline in Zone\n(all zones 2016–2025, deseasonalized)",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_cascade_total_mw.png", dpi=150)
print(f"Saved {FIG_DIR}/entsoe_cascade_total_mw.png")
plt.close()

# Per zone
print("\n--- Plot 4d: Total MW offline per zone ---")
cascade_zones = sorted(impacts_mw.zone.unique())
n_zones = len(cascade_zones)
fig, axes = plt.subplots(n_zones, 1, figsize=(14, 4.5 * n_zones), sharex=True)

for ax_i, zone in enumerate(cascade_zones):
    ax = axes[ax_i]
    zone_data = impacts_mw[impacts_mw.zone == zone]

    # Zone-specific quartiles
    zq25, zq50, zq75 = zone_data["total_mw_offline"].quantile([0.25, 0.5, 0.75])
    zone_mw_bins = [
        (0, zq25, f"< {zq25:.0f} MW"),
        (zq25, zq50, f"{zq25:.0f}–{zq50:.0f} MW"),
        (zq50, zq75, f"{zq50:.0f}–{zq75:.0f} MW"),
        (zq75, 999999, f"> {zq75:.0f} MW"),
    ]

    for (lo, hi, label), color in zip(zone_mw_bins, mw_colors):
        sub = zone_data[(zone_data.total_mw_offline >= lo) & (zone_data.total_mw_offline < hi)]
        if len(sub) < 10:
            continue
        means = [sub[f"change_{h}h"].mean() for h in HORIZONS]
        ax.plot(HORIZONS, means, marker="o", linewidth=2, markersize=5,
                color=color, label=f"{label} (n={len(sub):,})")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("EUR/MWh", fontsize=10)
    ax.set_title(f"{zone}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="best", ncol=2)

axes[-1].set_xlabel("Hours after outage start", fontsize=12)
axes[-1].set_xticks(HORIZONS)
axes[-1].set_xticklabels([str(h) for h in HORIZONS], fontsize=8)

fig.suptitle("Price Impact by Total MW Offline — Per Zone\n(deseasonalized, 2016–2025)",
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_cascade_total_mw_per_zone.png", dpi=150, bbox_inches="tight")
print(f"Saved {FIG_DIR}/entsoe_cascade_total_mw_per_zone.png")
plt.close()

# ── Compare: event count vs total MW — which predicts better? ────────────

print("\n--- Comparing event count vs total MW as predictors ---")

# Merge both into one df
compare = impacts_cascade[["zone", "event_start", "concurrent", "change_6h"]].copy()
compare = compare.merge(
    impacts_mw[["zone", "event_start", "total_mw_offline"]],
    on=["zone", "event_start"],
    how="inner"
)

from scipy import stats

# Correlation: concurrent count vs price change at +6h
valid = compare.dropna(subset=["change_6h"])
r_count, p_count = stats.pearsonr(valid["concurrent"], valid["change_6h"])
r_mw, p_mw = stats.pearsonr(valid["total_mw_offline"], valid["change_6h"])

print(f"\nCorrelation with price change at +6h:")
print(f"  Event count:    r={r_count:+.4f}, p={p_count:.4f}")
print(f"  Total MW:       r={r_mw:+.4f}, p={p_mw:.4f}")
print(f"  Better predictor: {'Total MW' if abs(r_mw) > abs(r_count) else 'Event count'}")

# Per zone
print(f"\nPer-zone correlation:")
print(f"  {'Zone':<6s} {'r(count)':>10s} {'r(MW)':>10s} {'Better':>12s}")
print(f"  {'-'*40}")
for zone in sorted(valid.zone.unique()):
    zv = valid[valid.zone == zone]
    if len(zv) < 30:
        continue
    rc, _ = stats.pearsonr(zv["concurrent"], zv["change_6h"])
    rm, _ = stats.pearsonr(zv["total_mw_offline"], zv["change_6h"])
    better = "MW" if abs(rm) > abs(rc) else "Count"
    print(f"  {zone:<6s} {rc:>+10.4f} {rm:>+10.4f} {better:>12s}")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 5: Seasonal patterns — bucket frequency by month
# ══════════════════════════════════════════════════════════════════════════

print("\n--- Plot 5: Seasonal patterns ---")

# Top 12 buckets by count
top12 = bucket_counts.head(12).index.tolist()
top12_data = filtered[filtered.bucket.isin(top12)]

# Monthly distribution
monthly = top12_data.groupby(["bucket", "month"]).size().unstack(fill_value=0)
# Normalize per bucket (% of that bucket's events per month)
monthly_pct = monthly.div(monthly.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(14, 8))

month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Heatmap
data = monthly_pct.reindex(columns=range(1, 13), fill_value=0).values
im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

ax.set_xticks(range(12))
ax.set_xticklabels(month_labels, fontsize=11)
ax.set_yticks(range(len(monthly_pct)))
ax.set_yticklabels([f"{b} (n={bucket_counts[b]:,})" for b in monthly_pct.index], fontsize=9)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        if val > 0:
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=7,
                    color="white" if val > 15 else "black")

plt.colorbar(im, ax=ax, label="% of bucket events", shrink=0.8)
ax.set_title("Seasonal Distribution of Outage Types\n(% of each bucket's events per month, 2016–2025)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_bucket_seasonality.png", dpi=150, bbox_inches="tight")
print(f"Saved {FIG_DIR}/entsoe_bucket_seasonality.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# PLOT 6: Training signal audit — which buckets carry predictive value?
# ══════════════════════════════════════════════════════════════════════════

print("\n--- Plot 6: Signal audit ---")

# For each bucket: mean |change| at +6h, statistical significance (t-test vs 0)
from scipy import stats

audit = []
for bucket in good_buckets:
    b_data = filtered[filtered.bucket == bucket]
    changes_6h = b_data["change_6h"].dropna()
    changes_1h = b_data["change_1h"].dropna()
    changes_24h = b_data["change_24h"].dropna()

    if len(changes_6h) < 10:
        continue

    t_stat, p_val = stats.ttest_1samp(changes_6h, 0)
    mean_abs = changes_6h.abs().mean()

    audit.append({
        "bucket": bucket,
        "n": len(changes_6h),
        "mean_1h": changes_1h.mean(),
        "mean_6h": changes_6h.mean(),
        "mean_24h": changes_24h.mean(),
        "mean_abs_6h": mean_abs,
        "t_stat": t_stat,
        "p_value": p_val,
        "significant": p_val < 0.05,
        "pct_unplanned": 100 * (b_data.unavailability_type == "Unplanned").mean(),
        "avg_mw": b_data.unavailable_mw.mean(),
    })

adf = pd.DataFrame(audit).sort_values("mean_abs_6h", ascending=False)

# Scatter: mean_abs_6h vs n, colored by significance
fig, ax = plt.subplots(figsize=(14, 8))

for _, row in adf.iterrows():
    color = "#E74C3C" if row.significant and row.mean_6h > 0 else \
            "#3498DB" if row.significant and row.mean_6h <= 0 else "#95A5A6"
    marker = "o" if row.significant else "x"
    size = max(30, min(300, row.n / 3))
    ax.scatter(row.n, row.mean_abs_6h, s=size, c=color, marker=marker,
               alpha=0.7, edgecolors="black", linewidth=0.5)
    ax.annotate(row.bucket, (row.n, row.mean_abs_6h),
                fontsize=7, ha="left", va="bottom",
                xytext=(5, 3), textcoords="offset points")

ax.set_xlabel("Number of events", fontsize=12)
ax.set_ylabel("Mean |price change| at +6h (EUR/MWh)", fontsize=12)
ax.legend(handles=[
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#E74C3C",
               markersize=10, label="Significant positive (p<0.05)"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498DB",
               markersize=10, label="Significant negative (p<0.05)"),
    plt.Line2D([0], [0], marker="x", color="#95A5A6",
               markersize=10, label="Not significant"),
], fontsize=10, loc="upper right")
ax.set_title("Training Signal Audit: Which Buckets Carry Predictive Value?\n"
             "(mean |price change| at +6h, deseasonalized, all zones 2016–2025)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_bucket_signal_audit.png", dpi=150, bbox_inches="tight")
print(f"Saved {FIG_DIR}/entsoe_bucket_signal_audit.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 110}")
print("TRAINING SIGNAL AUDIT — sorted by signal strength")
print(f"{'=' * 110}")
print(f"{'Bucket':<30s} {'n':>5s} {'%Unpl':>6s} {'AvgMW':>7s} "
      f"{'Δ+1h':>7s} {'Δ+6h':>7s} {'Δ+24h':>7s} {'|Δ|6h':>7s} "
      f"{'t-stat':>7s} {'p-val':>8s} {'Sig?':>5s}")
print("-" * 110)

for _, row in adf.iterrows():
    sig = "YES" if row.significant else "no"
    print(f"{row.bucket:<30s} {row.n:>5.0f} {row.pct_unplanned:>5.0f}% {row.avg_mw:>6.0f} "
          f"{row.mean_1h:>+6.2f} {row.mean_6h:>+6.2f} {row.mean_24h:>+6.2f} {row.mean_abs_6h:>6.2f} "
          f"{row.t_stat:>+6.2f} {row.p_value:>8.4f} {sig:>5s}")

# Key takeaways
sig_positive = adf[(adf.significant) & (adf.mean_6h > 0)]
sig_negative = adf[(adf.significant) & (adf.mean_6h <= 0)]
not_sig = adf[~adf.significant]

print(f"\n{'=' * 70}")
print("VERDICT")
print(f"{'=' * 70}")
print(f"Buckets with significant POSITIVE signal (include in text): {len(sig_positive)}")
for _, row in sig_positive.iterrows():
    print(f"  + {row.bucket}: +{row.mean_6h:.2f} EUR/MWh at +6h (p={row.p_value:.4f})")

print(f"\nBuckets with significant NEGATIVE signal (include in text): {len(sig_negative)}")
for _, row in sig_negative.iterrows():
    print(f"  - {row.bucket}: {row.mean_6h:.2f} EUR/MWh at +6h (p={row.p_value:.4f})")

print(f"\nBuckets with NO significant signal (consider excluding): {len(not_sig)}")
for _, row in not_sig.iterrows():
    print(f"  ~ {row.bucket}: {row.mean_6h:+.2f} EUR/MWh (p={row.p_value:.2f}, n={row.n:.0f})")
