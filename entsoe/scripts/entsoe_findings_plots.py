"""
ENTSO-E / Nord Pool UMM — Findings & Presentation Plots (v2)
=============================================================
Uses parsed UMM CSVs with proper Planned/Unplanned labels + parquet daily data.

Usage:
  uv run python scripts/entsoe_findings_plots.py
"""

import glob
import re
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)

PARQUET_DIR = os.path.join(_ROOT_DIR, "data/example_entsoe")
UMM_DIR = os.path.join(_ROOT_DIR, "data/example_entsoe/entsoe_examples_new")
FIG_DIR = os.path.join(_ROOT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════

# Load all daily parquets
pq_frames = []
for path in sorted(glob.glob(f"{PARQUET_DIR}/*.parquet")):
    if "training" in path:
        continue
    df = pd.read_parquet(path)
    name = os.path.basename(path).replace(".parquet", "")
    parts = name.rsplit("_", 1)
    df["zone"] = parts[0]
    df["year"] = int(parts[1])
    df["series_id"] = name
    pq_frames.append(df)

daily = pd.concat(pq_frames).reset_index(drop=True)
daily["date"] = pd.to_datetime(daily["t"])

# Load all parsed UMM CSVs
umm_frames = []
for folder in sorted(glob.glob(f"{UMM_DIR}/*")):
    name = os.path.basename(folder)
    parts = name.rsplit("_", 1)
    zone, year = parts[0], int(parts[1])
    csv_files = glob.glob(f"{folder}/umm_parsed_*.csv")
    if not csv_files:
        continue
    df = pd.read_csv(csv_files[0])
    df["zone"] = zone
    df["year"] = year
    umm_frames.append(df)

umm = pd.concat(umm_frames).reset_index(drop=True)
umm["event_start"] = pd.to_datetime(umm["event_start"], format="ISO8601", utc=True)
umm["event_end"] = pd.to_datetime(umm["event_end"], format="ISO8601", utc=True)
umm["event_date"] = umm["event_start"].dt.strftime("%Y-%m-%d")

print(f"Daily rows: {len(daily):,}")
print(f"UMM events: {len(umm):,}")
print(f"Unique reason texts: {umm.reason_text.nunique():,}")
print(f"Zones: {sorted(daily.zone.unique())}")
print(f"Years: {sorted(daily.year.unique())}")

# ── Build proper daily labels from UMM CSVs ───────────────────────────────

# For each series, mark days with unplanned/planned outages from parsed CSVs
daily_umm = []
for (zone, year), grp in umm.groupby(["zone", "year"]):
    year_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    year_end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")

    day_unplanned = {}  # date_str → total unplanned MW
    day_planned = {}    # date_str → total planned MW
    day_fuel_types = {} # date_str → set of fuel types
    day_n_events = {}   # date_str → count

    for _, row in grp.iterrows():
        evt_start = max(row["event_start"], year_start)
        evt_end = min(row["event_end"], year_end) if pd.notna(row["event_end"]) else year_end
        if evt_start >= evt_end:
            continue

        days = pd.date_range(evt_start.normalize(), evt_end.normalize(), freq="D")
        mw = row["unavailable_mw"] if pd.notna(row["unavailable_mw"]) else 0

        for d in days:
            ds = d.strftime("%Y-%m-%d")
            if row["unavailability_type"] == "Unplanned":
                day_unplanned[ds] = day_unplanned.get(ds, 0) + mw
            else:
                day_planned[ds] = day_planned.get(ds, 0) + mw
            day_n_events[ds] = day_n_events.get(ds, 0) + 1
            if pd.notna(row["fuel_type"]):
                if ds not in day_fuel_types:
                    day_fuel_types[ds] = set()
                day_fuel_types[ds].add(row["fuel_type"])

    all_dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    for d in all_dates:
        ds = d.strftime("%Y-%m-%d")
        daily_umm.append({
            "t": ds, "zone": zone, "year": year,
            "unplanned_mw": day_unplanned.get(ds, 0),
            "planned_mw": day_planned.get(ds, 0),
            "n_events": day_n_events.get(ds, 0),
            "fuel_types": ", ".join(sorted(day_fuel_types.get(ds, set()))),
        })

daily_umm = pd.DataFrame(daily_umm)

# Merge with daily prices
daily = daily.merge(daily_umm, on=["t", "zone", "year"], how="left")
daily["unplanned_mw"] = daily["unplanned_mw"].fillna(0)
daily["planned_mw"] = daily["planned_mw"].fillna(0)
daily["n_events"] = daily["n_events"].fillna(0).astype(int)
daily["total_unavail_mw"] = daily["unplanned_mw"] + daily["planned_mw"]
daily["has_text"] = daily["text"].fillna("") != ""
daily["has_unplanned"] = daily["unplanned_mw"] > 0
daily["planned_only"] = (daily["planned_mw"] > 0) & ~daily["has_unplanned"]
daily["no_outage"] = daily["n_events"] == 0

# Forward price changes per series
for sid, grp in daily.groupby("series_id"):
    idx = grp.index
    daily.loc[idx, "price_change_1d"] = grp["y_t"].shift(-1) - grp["y_t"]
    daily.loc[idx, "price_change_3d"] = grp["y_t"].shift(-3) - grp["y_t"]
    daily.loc[idx, "price_change_5d"] = grp["y_t"].shift(-5) - grp["y_t"]
    daily.loc[idx, "abs_change_1d"] = (grp["y_t"].shift(-1) - grp["y_t"]).abs()

n_unplanned = daily.has_unplanned.sum()
n_planned = daily.planned_only.sum()
n_none = daily.no_outage.sum()
print(f"\nDaily breakdown: {n_unplanned:,} unplanned, {n_planned:,} planned-only, {n_none:,} no outage")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 1: Coverage heatmap
# ══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 5))

zones = sorted(daily.zone.unique())
years = sorted(daily.year.unique())
coverage = pd.pivot_table(daily, values="has_text", index="zone", columns="year", aggfunc="mean")
coverage = coverage.reindex(zones).reindex(columns=years)

im = ax.imshow(coverage.values * 100, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
ax.set_xticks(range(len(years)))
ax.set_xticklabels(years, rotation=45)
ax.set_yticks(range(len(zones)))
ax.set_yticklabels(zones)

for i in range(len(zones)):
    for j in range(len(years)):
        val = coverage.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val*100:.0f}%", ha="center", va="center",
                    fontsize=8, color="black" if val > 0.4 else "white")

plt.colorbar(im, ax=ax, label="% days with text annotation", shrink=0.8)
ax.set_title("UMM Text Annotation Coverage by Zone and Year", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_coverage_heatmap.png", dpi=150)
print(f"Saved {FIG_DIR}/entsoe_coverage_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 2: Price change by outage type (from parsed labels)
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for i, (horizon, col) in enumerate([(1, "price_change_1d"), (3, "price_change_3d"), (5, "price_change_5d")]):
    ax = axes[i]
    categories = ["Unplanned\noutage", "Planned\nonly", "No\noutage"]
    masks = [daily.has_unplanned, daily.planned_only, daily.no_outage]
    means = [daily.loc[m, col].dropna().mean() for m in masks]
    sems = [daily.loc[m, col].dropna().sem() for m in masks]
    counts = [daily.loc[m, col].dropna().shape[0] for m in masks]
    colors = ["#E74C3C", "#F39C12", "#95A5A6"]

    bars = ax.bar(categories, means, yerr=sems, color=colors, edgecolor="white", capsize=5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("EUR/MWh" if i == 0 else "")
    ax.set_title(f"{horizon}-day forward\nprice change", fontsize=11)

    for bar, mean, n in zip(bars, means, counts):
        y_pos = bar.get_height() + (0.15 if mean >= 0 else -0.3)
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f"{mean:+.2f}\n(n={n:,})", ha="center", va="bottom", fontsize=8, fontweight="bold")

fig.suptitle("Average Forward Price Change by Outage Type\n(from parsed UMM labels, all zones 2016–2025)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_price_change_by_type.png", dpi=150)
print(f"Saved {FIG_DIR}/entsoe_price_change_by_type.png")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 3: Price volatility by outage type
# ══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 5))

categories = ["Unplanned outage", "Planned only", "No outage"]
masks = [daily.has_unplanned, daily.planned_only, daily.no_outage]
colors = ["#E74C3C", "#F39C12", "#95A5A6"]

vol_data = [daily.loc[m, "abs_change_1d"].dropna() for m in masks]
bp = ax.boxplot(vol_data, tick_labels=categories, patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

means = [v.mean() for v in vol_data]
for i, m in enumerate(means):
    ax.plot(i + 1, m, "D", color="black", markersize=8, zorder=5)
    ax.text(i + 1.15, m, f"mean={m:.1f}", va="center", fontsize=9)

ax.set_ylabel("Absolute 1-day price change (EUR/MWh)")
ax.set_title("Price Volatility by Outage Type\n(parsed UMM labels, all zones 2016–2025)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_volatility_by_type.png", dpi=150)
print(f"Saved {FIG_DIR}/entsoe_volatility_by_type.png")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 4: Unplanned MW vs price — scatter with trend
# ══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 6))

valid = daily[daily.unplanned_mw > 0].copy()

zone_colors = {"NO1": "#3498DB", "NO2": "#2980B9", "FI": "#E74C3C",
               "DK1": "#2ECC71", "DK2": "#27AE60", "SE1": "#F39C12", "SE3": "#E67E22"}

for zone in sorted(valid.zone.unique()):
    z = valid[valid.zone == zone]
    ax.scatter(z.unplanned_mw, z.y_t, alpha=0.15, s=15,
               color=zone_colors.get(zone, "#999"), label=zone)

# Binned trend
bins = pd.cut(valid.unplanned_mw, bins=15)
trend = valid.groupby(bins, observed=True)["y_t"].mean()
bin_centers = [(b.left + b.right) / 2 for b in trend.index]
ax.plot(bin_centers, trend.values, "k-", linewidth=2.5, label="Binned mean", zorder=10)

corr = valid["unplanned_mw"].corr(valid["y_t"])
ax.text(0.02, 0.98, f"Pearson r = {corr:.3f}\nn = {len(valid):,}",
        transform=ax.transAxes, va="top", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

ax.set_xlabel("Unplanned Unavailable Capacity (MW)")
ax.set_ylabel("Day-ahead Price (EUR/MWh)")
ax.set_title("Unplanned Outage Capacity vs Day-ahead Price\n(all zones, 2016–2025)",
             fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_unplanned_vs_price.png", dpi=150)
print(f"Saved {FIG_DIR}/entsoe_unplanned_vs_price.png")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 5: SE1 2025 showcase — price + outage MW stacked
# ══════════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]})

showcase = daily[(daily.zone == "SE1") & (daily.year == 2025)].copy()
dates = showcase["date"]

ax1.plot(dates, showcase["y_t"], lw=1.2, color="#2C3E50", label="Day-ahead price")
ax1.fill_between(dates, 0, showcase["y_t"],
                 where=showcase["has_unplanned"], alpha=0.3, color="#E74C3C",
                 label="Unplanned outage day")
ax1.fill_between(dates, 0, showcase["y_t"],
                 where=showcase["planned_only"], alpha=0.15, color="#F39C12",
                 label="Planned only")
ax1.set_ylabel("EUR/MWh")
ax1.set_title("SE1 (Sweden North) 2025 — Prices with UMM Outage Annotations",
              fontsize=13, fontweight="bold")
ax1.legend(loc="upper left", fontsize=9)

# Stacked MW bar
ax2.bar(dates, showcase["unplanned_mw"], color="#E74C3C", alpha=0.7, label="Unplanned MW")
ax2.bar(dates, showcase["planned_mw"], bottom=showcase["unplanned_mw"],
        color="#F39C12", alpha=0.5, label="Planned MW")
ax2.set_ylabel("Unavailable MW")
ax2.legend(loc="upper left", fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_se1_2025_showcase.png", dpi=150)
print(f"Saved {FIG_DIR}/entsoe_se1_2025_showcase.png")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 6: Signal by zone
# ══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 5))

zone_signal = []
for zone in sorted(daily.zone.unique()):
    z = daily[daily.zone == zone]
    unplanned_mean = z.loc[z.has_unplanned, "price_change_3d"].dropna().mean()
    baseline_mean = z.loc[~z.has_unplanned, "price_change_3d"].dropna().mean()
    diff = unplanned_mean - baseline_mean
    n = z.has_unplanned.sum()
    zone_signal.append({"zone": zone, "diff": diff, "n_unplanned": n})

zdf = pd.DataFrame(zone_signal).sort_values("diff", ascending=True)

colors = ["#E74C3C" if d > 0 else "#3498DB" for d in zdf["diff"]]
bars = ax.barh(zdf["zone"], zdf["diff"], color=colors, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("3-day price change: Unplanned − Baseline (EUR/MWh)")
ax.set_title("Unplanned Outage Price Impact by Zone\n(excess 3-day forward return vs non-unplanned days)",
             fontsize=13, fontweight="bold")

for bar, val, n in zip(bars, zdf["diff"], zdf["n_unplanned"]):
    x = bar.get_width() + 0.1 * np.sign(bar.get_width())
    ax.text(x, bar.get_y() + bar.get_height()/2,
            f"{val:+.2f} (n={n:,})", va="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_signal_by_zone.png", dpi=150)
print(f"Saved {FIG_DIR}/entsoe_signal_by_zone.png")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 7: Fuel type breakdown of outages
# ══════════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 7a: Count by fuel type
fuel_counts = umm.groupby(["fuel_type", "unavailability_type"]).size().unstack(fill_value=0)
fuel_counts["total"] = fuel_counts.sum(axis=1)
fuel_counts = fuel_counts.sort_values("total", ascending=True).tail(10)

fuel_counts[["Planned", "Unplanned"]].plot.barh(
    ax=ax1, stacked=True, color=["#F39C12", "#E74C3C"], edgecolor="white")
ax1.set_xlabel("Number of outage events")
ax1.set_title("Top 10 Fuel Types by Outage Count", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)

# 7b: Average price on unplanned outage days by fuel type
fuel_price = []
for ft in umm.fuel_type.dropna().unique():
    ft_events = umm[(umm.fuel_type == ft) & (umm.unavailability_type == "Unplanned")]
    ft_dates = set(ft_events["event_date"])
    if len(ft_dates) < 10:
        continue
    matching = daily[daily.t.isin(ft_dates)]
    if len(matching) > 0:
        fuel_price.append({
            "fuel_type": ft,
            "avg_price": matching["y_t"].mean(),
            "n_days": len(ft_dates),
        })

fpdf = pd.DataFrame(fuel_price).sort_values("avg_price", ascending=True)
ax2.barh(fpdf.fuel_type, fpdf.avg_price, color="#E74C3C", edgecolor="white", alpha=0.7)
ax2.set_xlabel("Average day-ahead price (EUR/MWh)")
ax2.set_title("Avg Price on Unplanned Outage Days\nby Fuel Type", fontsize=12, fontweight="bold")
for _, r in fpdf.iterrows():
    ax2.text(r.avg_price + 0.5, r.fuel_type, f"{r.avg_price:.1f} (n={r.n_days})",
             va="center", fontsize=9)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_fuel_type_breakdown.png", dpi=150)
print(f"Saved {FIG_DIR}/entsoe_fuel_type_breakdown.png")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 8: Dataset summary
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 8a: Rows per zone
zone_counts = daily.groupby("zone").size()
axes[0].bar(zone_counts.index, zone_counts.values, color="#3498DB", edgecolor="white")
axes[0].set_ylabel("Total daily rows")
axes[0].set_title("Rows per Zone")
for i, (z, v) in enumerate(zone_counts.items()):
    axes[0].text(i, v + 20, str(v), ha="center", fontsize=9)

# 8b: Annotation rate per zone
ann_rate = daily.groupby("zone")["has_text"].mean() * 100
axes[1].bar(ann_rate.index, ann_rate.values, color="#2ECC71", edgecolor="white")
axes[1].set_ylabel("% days annotated")
axes[1].set_title("Annotation Coverage")
axes[1].set_ylim(0, 110)
for i, (z, v) in enumerate(ann_rate.items()):
    axes[1].text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9)

# 8c: Unplanned vs planned split
unplanned_rate = daily.groupby("zone")["has_unplanned"].mean() * 100
planned_rate = daily.groupby("zone")["planned_only"].mean() * 100
x = np.arange(len(zones))
w = 0.4
axes[2].bar(x - w/2, unplanned_rate.reindex(zones).values, w,
            color="#E74C3C", edgecolor="white", label="Has unplanned")
axes[2].bar(x + w/2, planned_rate.reindex(zones).values, w,
            color="#F39C12", edgecolor="white", label="Planned only")
axes[2].set_xticks(x)
axes[2].set_xticklabels(zones)
axes[2].set_ylabel("% of days")
axes[2].set_title("Outage Type Split")
axes[2].legend(fontsize=9)

fig.suptitle(f"Dataset Summary — {len(daily):,} rows, {daily.series_id.nunique()} series, "
             f"{sorted(daily.year.unique())[0]}–{sorted(daily.year.unique())[-1]}",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_dataset_summary.png", dpi=150)
print(f"Saved {FIG_DIR}/entsoe_dataset_summary.png")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 9: Energy crisis signal — 2021-2022 vs other years
# ══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 5))

yearly_signal = []
for year in sorted(daily.year.unique()):
    y = daily[daily.year == year]
    unplanned_mean = y.loc[y.has_unplanned, "price_change_3d"].dropna().mean()
    baseline_mean = y.loc[~y.has_unplanned, "price_change_3d"].dropna().mean()
    diff = unplanned_mean - baseline_mean
    avg_price = y.y_t.mean()
    yearly_signal.append({"year": year, "diff": diff, "avg_price": avg_price})

ydf = pd.DataFrame(yearly_signal)

color_map = {2021: "#E74C3C", 2022: "#E74C3C"}  # crisis years highlighted
bar_colors = [color_map.get(y, "#3498DB") for y in ydf.year]

bars = ax.bar(ydf.year, ydf["diff"], color=bar_colors, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_xlabel("Year")
ax.set_ylabel("Excess 3-day price change (EUR/MWh)")
ax.set_title("Unplanned Outage Signal Strength by Year\n(red = energy crisis years)",
             fontsize=13, fontweight="bold")
ax.set_xticks(ydf.year)

for bar, val, price in zip(bars, ydf["diff"], ydf["avg_price"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{val:+.2f}\n(avg €{price:.0f})", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/entsoe_yearly_signal.png", dpi=150)
print(f"Saved {FIG_DIR}/entsoe_yearly_signal.png")

# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SUMMARY FOR PRESENTATION")
print("=" * 70)
print(f"\nDataset: {len(daily):,} daily rows, {daily.series_id.nunique()} zone-year series")
print(f"UMM events: {len(umm):,} ({umm.reason_text.nunique():,} unique reason texts)")
print(f"Zones: {', '.join(sorted(daily.zone.unique()))}")
print(f"Years: {sorted(daily.year.unique())[0]}–{sorted(daily.year.unique())[-1]}")
print(f"Annotated rows: {daily.has_text.sum():,} ({100*daily.has_text.mean():.0f}%)")

print(f"\nOutage breakdown (from parsed labels):")
print(f"  Unplanned outage days: {daily.has_unplanned.sum():,}")
print(f"  Planned-only days:     {daily.planned_only.sum():,}")
print(f"  No outage days:        {daily.no_outage.sum():,}")

print(f"\n3-day forward price change by outage type:")
for label, mask in [("Unplanned outage", daily.has_unplanned),
                     ("Planned only", daily.planned_only),
                     ("No outage", daily.no_outage)]:
    vals = daily.loc[mask, "price_change_3d"].dropna()
    print(f"  {label:20s}: {vals.mean():+.2f} EUR/MWh  (n={len(vals):,})")

print(f"\nPlots saved to {FIG_DIR}/entsoe_*.png")
