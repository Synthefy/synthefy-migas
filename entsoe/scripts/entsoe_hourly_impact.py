"""
Hourly price impact analysis after outage events
=================================================
Measures price change at 1h, 3h, 6h, 12h, 24h after outage start.

Usage:
  uv run python scripts/entsoe_hourly_impact.py
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

# ── Load all hourly prices ────────────────────────────────────────────────

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
    df["year"] = year
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    price_frames.append(df)

prices = pd.concat(price_frames).reset_index(drop=True)
print(f"Hourly price rows: {len(prices):,}")

# ── Load all UMM events ──────────────────────────────────────────────────

umm_frames = []
for folder in sorted(glob.glob(f"{DATA_DIR}/*")):
    name = os.path.basename(folder)
    parts = name.rsplit("_", 1)
    zone, year = parts[0], int(parts[1])

    umm_csv = glob.glob(f"{folder}/umm_parsed_*.csv")
    if not umm_csv:
        continue
    df = pd.read_csv(umm_csv[0])
    df["zone"] = zone
    df["year"] = year
    umm_frames.append(df)

umm = pd.concat(umm_frames).reset_index(drop=True)
umm["event_start"] = pd.to_datetime(umm["event_start"], format="ISO8601", utc=True)
umm["event_start_hour"] = umm["event_start"].dt.floor("h")

print(f"UMM events: {len(umm):,}")

# ── Categorize reasons ────────────────────────────────────────────────────

def categorize_reason(text):
    if pd.isna(text):
        return "Unknown"
    t = str(text).lower().strip()
    if any(w in t for w in ["ice", "icing", "cold", "frost", "snow", "weather", "storm", "flood"]):
        return "Weather/Icing"
    if any(w in t for w in ["trip", "tripped"]):
        return "Tripped"
    if any(w in t for w in ["failure", "fault", "faliure", "broken", "defect", "malfunction", "problem", "error"]):
        return "Failure/Fault"
    if any(w in t for w in ["maintenance", "maintenace", "maitenance", "manintenance"]):
        return "Maintenance"
    if any(w in t for w in ["repair", "fix", "replace"]):
        return "Repair"
    if any(w in t for w in ["test", "commission", "trial"]):
        return "Testing"
    if any(w in t for w in ["overhaul", "refurbish", "renovation", "upgrade"]):
        return "Overhaul/Upgrade"
    if any(w in t for w in ["grid", "tso", "network", "transmission"]):
        return "Grid issues"
    if any(w in t for w in ["inspection"]):
        return "Inspection"
    if any(w in t for w in ["unknown"]):
        return "Unknown"
    return "Other"

umm["reason_category"] = umm["reason_text"].apply(categorize_reason)

# ── Compute hourly price changes after each event ─────────────────────────

HORIZONS = [1, 3, 6, 12, 24]
WINDOW = 2  # ±2 hours rolling average around each horizon

# ── Build hourly seasonal profiles per zone ───────────────────────────────
print("Building hourly seasonal profiles per zone...")
seasonal_profiles = {}
for zone in prices.zone.unique():
    zp = prices[prices.zone == zone].copy()
    zp["hour"] = zp.timestamp_utc.dt.hour
    profile = zp.groupby("hour")["price_eur_mwh"].mean()
    seasonal_profiles[zone] = profile
    print(f"  {zone}: min={profile.min():.1f} (h={profile.idxmin()}), max={profile.max():.1f} (h={profile.idxmax()})")


def compute_event_impacts(events_df, prices_df, label=""):
    """Deseasonalized ±W rolling avg: removes daily price cycle."""
    results = []

    for zone in events_df.zone.unique():
        zone_prices = prices_df[prices_df.zone == zone].set_index("timestamp_utc")["price_eur_mwh"].sort_index()
        zone_events = events_df[events_df.zone == zone]
        profile = seasonal_profiles.get(zone)

        for _, evt in zone_events.iterrows():
            t0 = evt["event_start_hour"]
            if t0 not in zone_prices.index:
                continue

            t0_window = zone_prices.loc[t0 - pd.Timedelta(hours=WINDOW):t0 + pd.Timedelta(hours=WINDOW)]
            if t0_window.empty:
                continue
            p0 = t0_window.mean()
            h0 = t0.hour

            row = {
                "zone": zone,
                "event_start": t0,
                "price_at_event": p0,
                "unavailability_type": evt["unavailability_type"],
                "reason_category": evt["reason_category"],
                "fuel_type": evt.get("fuel_type", ""),
                "unavailable_mw": evt.get("unavailable_mw", 0),
            }

            for h in HORIZONS:
                th = t0 + pd.Timedelta(hours=h)
                th_window = zone_prices.loc[th - pd.Timedelta(hours=WINDOW):th + pd.Timedelta(hours=WINDOW)]
                if not th_window.empty:
                    raw_change = th_window.mean() - p0
                    seasonal_change = profile.loc[th.hour] - profile.loc[h0]
                    row[f"change_{h}h"] = raw_change - seasonal_change
                else:
                    row[f"change_{h}h"] = np.nan

            results.append(row)

    return pd.DataFrame(results)


def compute_event_impacts_single(events_df, prices_df):
    """Deseasonalized single-point: removes daily price cycle."""
    results = []

    for zone in events_df.zone.unique():
        zone_prices = prices_df[prices_df.zone == zone].set_index("timestamp_utc")["price_eur_mwh"].sort_index()
        zone_events = events_df[events_df.zone == zone]
        profile = seasonal_profiles.get(zone)

        for _, evt in zone_events.iterrows():
            t0 = evt["event_start_hour"]
            if t0 not in zone_prices.index:
                continue
            p0 = zone_prices.loc[t0]
            h0 = t0.hour

            row = {
                "zone": zone,
                "event_start": t0,
                "price_at_event": p0,
                "unavailability_type": evt["unavailability_type"],
                "reason_category": evt["reason_category"],
                "fuel_type": evt.get("fuel_type", ""),
                "unavailable_mw": evt.get("unavailable_mw", 0),
            }

            for h in HORIZONS:
                th = t0 + pd.Timedelta(hours=h)
                if th in zone_prices.index:
                    raw_change = zone_prices.loc[th] - p0
                    seasonal_change = profile.loc[th.hour] - profile.loc[h0]
                    row[f"change_{h}h"] = raw_change - seasonal_change
                else:
                    row[f"change_{h}h"] = np.nan

            results.append(row)

    return pd.DataFrame(results)


print("Computing hourly impacts — rolling average (±2h)...")
impacts = compute_event_impacts(umm, prices)

print("Computing hourly impacts — single point...")
impacts_single = compute_event_impacts_single(umm, prices)
print(f"Events with price match: {len(impacts):,}")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 1: Boxplot — price change at different horizons, unplanned vs planned
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, len(HORIZONS), figsize=(18, 6), sharey=True)

for i, h in enumerate(HORIZONS):
    ax = axes[i]
    col = f"change_{h}h"

    unplanned = impacts.loc[impacts.unavailability_type == "Unplanned", col].dropna()
    planned = impacts.loc[impacts.unavailability_type == "Planned", col].dropna()

    bp = ax.boxplot([unplanned, planned], tick_labels=["Unplanned", "Planned"],
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor("#E74C3C")
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("#F39C12")
    bp["boxes"][1].set_alpha(0.7)

    # Mark means
    for j, data in enumerate([unplanned, planned]):
        ax.plot(j + 1, data.mean(), "D", color="black", markersize=7, zorder=5)
        ax.text(j + 1.2, data.mean(), f"{data.mean():+.2f}", va="center", fontsize=8)

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(f"+{h}h", fontsize=12, fontweight="bold")
    if i == 0:
        ax.set_ylabel("Price change (EUR/MWh)")

fig.suptitle("Price Change After Outage Start — Unplanned vs Planned\n(hourly resolution, all zones 2016–2025)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_hourly_boxplot_type.png", dpi=150)
print(f"Saved {FIG_DIR}/entsoe_hourly_boxplot_type.png")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 1b: Boxplots per zone — unplanned only, all horizons
# ══════════════════════════════════════════════════════════════════════════

zones = sorted(impacts.zone.unique())
fig, axes = plt.subplots(len(zones), len(HORIZONS), figsize=(20, 3.5 * len(zones)), sharey="row")

for row_i, zone in enumerate(zones):
    zone_data = impacts[(impacts.zone == zone) & (impacts.unavailability_type == "Unplanned")]
    zone_planned = impacts[(impacts.zone == zone) & (impacts.unavailability_type == "Planned")]

    for col_i, h in enumerate(HORIZONS):
        ax = axes[row_i, col_i]
        col = f"change_{h}h"

        u = zone_data[col].dropna()
        p = zone_planned[col].dropna()

        if len(u) < 5 and len(p) < 5:
            ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center", va="center")
            continue

        data = [u, p]
        labels = [f"U (n={len(u)})", f"P (n={len(p)})"]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.5))
        bp["boxes"][0].set_facecolor("#E74C3C")
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor("#F39C12")
        bp["boxes"][1].set_alpha(0.7)

        for j, d in enumerate(data):
            if len(d) > 0:
                ax.plot(j + 1, d.mean(), "D", color="black", markersize=5, zorder=5)
                ax.text(j + 1.15, d.mean(), f"{d.mean():+.1f}", va="center", fontsize=7)

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

        if row_i == 0:
            ax.set_title(f"+{h}h", fontsize=11, fontweight="bold")
        if col_i == 0:
            ax.set_ylabel(f"{zone}\nEUR/MWh", fontsize=10, fontweight="bold")

fig.suptitle("Price Change After Outage Start — Per Zone\n(±2h rolling avg, 2016–2025)",
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_hourly_boxplot_per_zone.png", dpi=150)
print(f"Saved {FIG_DIR}/entsoe_hourly_boxplot_per_zone.png")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 1c: Single-point boxplot — all zones combined
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, len(HORIZONS), figsize=(18, 6), sharey=True)

for i, h in enumerate(HORIZONS):
    ax = axes[i]
    col = f"change_{h}h"

    unplanned = impacts_single.loc[impacts_single.unavailability_type == "Unplanned", col].dropna()
    planned = impacts_single.loc[impacts_single.unavailability_type == "Planned", col].dropna()

    bp = ax.boxplot([unplanned, planned], tick_labels=["Unplanned", "Planned"],
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor("#E74C3C")
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("#F39C12")
    bp["boxes"][1].set_alpha(0.7)

    for j, data in enumerate([unplanned, planned]):
        ax.plot(j + 1, data.mean(), "D", color="black", markersize=7, zorder=5)
        ax.text(j + 1.2, data.mean(), f"{data.mean():+.2f}", va="center", fontsize=8)

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(f"+{h}h", fontsize=12, fontweight="bold")
    if i == 0:
        ax.set_ylabel("Price change (EUR/MWh)")

fig.suptitle("Price Change After Outage Start — Single Point\n(price[t+N] − price[t], all zones 2016–2025)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_hourly_boxplot_type_single.png", dpi=150)
print(f"Saved {FIG_DIR}/entsoe_hourly_boxplot_type_single.png")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 1d: Single-point boxplots per zone
# ══════════════════════════════════════════════════════════════════════════

zones = sorted(impacts_single.zone.unique())
fig, axes = plt.subplots(len(zones), len(HORIZONS), figsize=(20, 3.5 * len(zones)), sharey="row")

for row_i, zone in enumerate(zones):
    zone_data = impacts_single[(impacts_single.zone == zone) & (impacts_single.unavailability_type == "Unplanned")]
    zone_planned = impacts_single[(impacts_single.zone == zone) & (impacts_single.unavailability_type == "Planned")]

    for col_i, h in enumerate(HORIZONS):
        ax = axes[row_i, col_i]
        col = f"change_{h}h"

        u = zone_data[col].dropna()
        p = zone_planned[col].dropna()

        if len(u) < 5 and len(p) < 5:
            ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center", va="center")
            continue

        data = [u, p]
        labels = [f"U (n={len(u)})", f"P (n={len(p)})"]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.5))
        bp["boxes"][0].set_facecolor("#E74C3C")
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor("#F39C12")
        bp["boxes"][1].set_alpha(0.7)

        for j, d in enumerate(data):
            if len(d) > 0:
                ax.plot(j + 1, d.mean(), "D", color="black", markersize=5, zorder=5)
                ax.text(j + 1.15, d.mean(), f"{d.mean():+.1f}", va="center", fontsize=7)

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

        if row_i == 0:
            ax.set_title(f"+{h}h", fontsize=11, fontweight="bold")
        if col_i == 0:
            ax.set_ylabel(f"{zone}\nEUR/MWh", fontsize=10, fontweight="bold")

fig.suptitle("Price Change After Outage Start — Per Zone, Single Point\n(price[t+N] − price[t], 2016–2025)",
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_hourly_boxplot_per_zone_single.png", dpi=150)
print(f"Saved {FIG_DIR}/entsoe_hourly_boxplot_per_zone_single.png")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 2: Mean price change trajectory — unplanned vs planned
# ══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 6))

for utype, color, marker in [("Unplanned", "#E74C3C", "o"), ("Planned", "#F39C12", "s")]:
    subset = impacts[impacts.unavailability_type == utype]
    means = [subset[f"change_{h}h"].mean() for h in HORIZONS]
    sems = [subset[f"change_{h}h"].sem() for h in HORIZONS]
    n = len(subset)

    ax.errorbar(HORIZONS, means, yerr=sems, color=color, marker=marker,
                linewidth=2, markersize=8, capsize=5, label=f"{utype} (n={n:,})")

    for h, m in zip(HORIZONS, means):
        ax.text(h + 0.3, m, f"{m:+.2f}", fontsize=8, color=color, fontweight="bold")

ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_xlabel("Hours after outage start")
ax.set_ylabel("Average price change (EUR/MWh)")
ax.set_xticks(HORIZONS)
ax.set_xticklabels([f"+{h}h" for h in HORIZONS])
ax.set_title("Price Trajectory After Outage Start\n(all zones, 2016–2025)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_hourly_trajectory.png", dpi=150)
print(f"Saved {FIG_DIR}/entsoe_hourly_trajectory.png")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 3: Trajectory by reason category (unplanned only)
# ══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 6))

unplanned = impacts[impacts.unavailability_type == "Unplanned"]

cat_colors = {
    "Weather/Icing": "#3498DB",
    "Tripped": "#E74C3C",
    "Failure/Fault": "#E67E22",
    "Grid issues": "#9B59B6",
    "Maintenance": "#95A5A6",
}

for cat, color in cat_colors.items():
    subset = unplanned[unplanned.reason_category == cat]
    if len(subset) < 50:
        continue
    means = [subset[f"change_{h}h"].mean() for h in HORIZONS]
    n = len(subset)
    ax.plot(HORIZONS, means, marker="o", linewidth=2, markersize=7,
            color=color, label=f"{cat} (n={n:,})")

ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_xlabel("Hours after outage start")
ax.set_ylabel("Average price change (EUR/MWh)")
ax.set_xticks(HORIZONS)
ax.set_xticklabels([f"+{h}h" for h in HORIZONS])
ax.set_title("Price Trajectory by Reason Category\n(unplanned outages only, all zones 2016–2025)",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_hourly_by_reason.png", dpi=150)
print(f"Saved {FIG_DIR}/entsoe_hourly_by_reason.png")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 4: Trajectory by fuel type (unplanned only)
# ══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 6))

fuel_colors = {
    "Nuclear": "#E74C3C",
    "Reservoir hydro": "#3498DB",
    "Biomass/CHP": "#2ECC71",
    "Coal/Hard coal": "#7F8C8D",
    "Natural Gas (CCGT)": "#F39C12",
}

for ft, color in fuel_colors.items():
    subset = unplanned[unplanned.fuel_type == ft]
    if len(subset) < 50:
        continue
    means = [subset[f"change_{h}h"].mean() for h in HORIZONS]
    n = len(subset)
    avg_mw = subset.unavailable_mw.mean()
    ax.plot(HORIZONS, means, marker="o", linewidth=2, markersize=7,
            color=color, label=f"{ft} (n={n:,}, ~{avg_mw:.0f}MW)")

ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_xlabel("Hours after outage start")
ax.set_ylabel("Average price change (EUR/MWh)")
ax.set_xticks(HORIZONS)
ax.set_xticklabels([f"+{h}h" for h in HORIZONS])
ax.set_title("Price Trajectory by Fuel Type\n(unplanned outages only, all zones 2016–2025)",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_hourly_by_fuel.png", dpi=150)
print(f"Saved {FIG_DIR}/entsoe_hourly_by_fuel.png")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 5: Severity bins — hourly trajectory
# ══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 6))

severity_bins = [
    (0, 100, "< 100 MW", "#95A5A6"),
    (100, 300, "100–300 MW", "#F39C12"),
    (300, 600, "300–600 MW", "#E67E22"),
    (600, 99999, "600+ MW", "#E74C3C"),
]

for lo, hi, label, color in severity_bins:
    subset = unplanned[(unplanned.unavailable_mw >= lo) & (unplanned.unavailable_mw < hi)]
    if len(subset) < 30:
        continue
    means = [subset[f"change_{h}h"].mean() for h in HORIZONS]
    n = len(subset)
    ax.plot(HORIZONS, means, marker="o", linewidth=2, markersize=7,
            color=color, label=f"{label} (n={n:,})")

ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_xlabel("Hours after outage start")
ax.set_ylabel("Average price change (EUR/MWh)")
ax.set_xticks(HORIZONS)
ax.set_xticklabels([f"+{h}h" for h in HORIZONS])
ax.set_title("Price Trajectory by Outage Severity\n(unplanned outages only, all zones 2016–2025)",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_hourly_by_severity.png", dpi=150)
print(f"Saved {FIG_DIR}/entsoe_hourly_by_severity.png")

# ── Summary ───────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("HOURLY IMPACT SUMMARY")
print("=" * 70)
print(f"\nEvents analyzed: {len(impacts):,}")
for utype in ["Unplanned", "Planned"]:
    subset = impacts[impacts.unavailability_type == utype]
    print(f"\n{utype} (n={len(subset):,}):")
    for h in HORIZONS:
        col = f"change_{h}h"
        m = subset[col].mean()
        s = subset[col].sem()
        print(f"  +{h:2d}h: {m:+.2f} ± {s:.2f} EUR/MWh")
