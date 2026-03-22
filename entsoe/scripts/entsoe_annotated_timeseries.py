"""
Annotated time series plot — shows prices with outage events and reason text.

Usage:
  uv run python scripts/entsoe_annotated_timeseries.py
"""

import os
import textwrap

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)

FIG_DIR = os.path.join(_ROOT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Zone → local timezone mapping ────────────────────────────────────────

ZONE_TZ = {
    "NO1": "Europe/Oslo",
    "NO2": "Europe/Oslo",
    "SE1": "Europe/Stockholm",
    "SE3": "Europe/Stockholm",
    "DK1": "Europe/Copenhagen",
    "DK2": "Europe/Copenhagen",
    "FI":  "Europe/Helsinki",
}

NIGHT_START_LOCAL = 21
NIGHT_END_LOCAL = 6


def shade_night(ax, t_start, t_end, zone):
    """Shade nighttime hours (21:00-06:00 local) on a UTC-indexed axis."""
    import pytz
    tz = pytz.timezone(ZONE_TZ.get(zone, "Europe/Oslo"))

    current = t_start.normalize()
    while current < t_end + pd.Timedelta(days=1):
        night_start_local = tz.localize(
            pd.Timestamp(current.year, current.month, current.day, NIGHT_START_LOCAL)
        )
        night_end_local = night_start_local + pd.Timedelta(hours=9)

        ns_utc = night_start_local.astimezone(pytz.UTC)
        ne_utc = night_end_local.astimezone(pytz.UTC)

        ns_clipped = max(ns_utc, t_start)
        ne_clipped = min(ne_utc, t_end)

        if ns_clipped < ne_clipped:
            ax.axvspan(ns_clipped, ne_clipped, alpha=0.06, color="#1a1a2e", zorder=0)

        current += pd.Timedelta(days=1)


# ── Config ────────────────────────────────────────────────────────────────

SHOWCASE = [
    # (zone, year, month_start, month_end, title_note)
    ("FI", 2022, 6, 7, "Nuclear scram + energy crisis"),
    ("FI", 2022, 11, 12, "Coal mill failures cascade"),
    ("SE1", 2025, 11, 12, "Winter failures"),
]

DATA_DIR = os.path.join(_ROOT_DIR, "data/entsoe_examples_new")


def make_annotated_plot(zone, year, month_start, month_end, title_note):
    """Create an annotated price + outage plot for a specific window."""

    # Load hourly prices
    prices = pd.read_csv(f"{DATA_DIR}/{zone}_{year}/entsoe_prices_{zone}.csv")
    prices["timestamp_utc"] = pd.to_datetime(prices["timestamp_utc"], utc=True)
    prices = prices.sort_values("timestamp_utc")

    # Load UMM events
    umm = pd.read_csv(f"{DATA_DIR}/{zone}_{year}/umm_parsed_{zone}_{year}.csv")
    umm["event_start"] = pd.to_datetime(umm["event_start"], format="ISO8601", utc=True)
    umm["event_end"] = pd.to_datetime(umm["event_end"], format="ISO8601", utc=True)

    # Filter to window
    t_start = pd.Timestamp(f"{year}-{month_start:02d}-01", tz="UTC")
    t_end = pd.Timestamp(f"{year}-{month_end:02d}-01", tz="UTC") + pd.DateOffset(months=1)

    p = prices[(prices.timestamp_utc >= t_start) & (prices.timestamp_utc < t_end)].copy()
    events = umm[(umm.event_start >= t_start) & (umm.event_start < t_end)].copy()

    # Sort events by MW for annotation priority
    events = events.sort_values("unavailable_mw", ascending=False)

    # ── Plot ──────────────────────────────────────────────────────────────

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(18, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    # Top: price line
    shade_night(ax1, t_start, t_end, zone)
    shade_night(ax2, t_start, t_end, zone)

    ax1.plot(p.timestamp_utc, p.price_eur_mwh, lw=0.8, color="#2C3E50", alpha=0.9)
    ax1.fill_between(p.timestamp_utc, 0, p.price_eur_mwh, alpha=0.05, color="#2C3E50")

    # Shade outage periods
    for _, evt in events.iterrows():
        evt_end = min(evt.event_end, t_end) if pd.notna(evt.event_end) else t_end
        color = "#E74C3C" if evt.unavailability_type == "Unplanned" else "#F39C12"
        alpha = 0.08
        ax1.axvspan(evt.event_start, evt_end, alpha=alpha, color=color, linewidth=0)

    # Annotate top events with text
    # Deduplicate: keep highest MW per day to avoid overlap
    events["event_day"] = events.event_start.dt.date
    top_events = events.drop_duplicates(subset=["event_day"], keep="first")

    # Pick top N events to annotate (avoid clutter)
    n_annotate = min(12, len(top_events))
    annotate_events = top_events.head(n_annotate)

    # Alternate text position above/below to reduce overlap
    y_max = p.price_eur_mwh.max()
    y_min = p.price_eur_mwh.min()
    y_range = y_max - y_min

    for i, (_, evt) in enumerate(annotate_events.iterrows()):
        # Find price at event time
        closest_idx = (p.timestamp_utc - evt.event_start).abs().idxmin()
        price_at_event = p.loc[closest_idx, "price_eur_mwh"]

        # Build label
        mw = f"{evt.unavailable_mw:.0f}MW" if pd.notna(evt.unavailable_mw) else ""
        reason = str(evt.reason_text)[:50] if pd.notna(evt.reason_text) else ""
        utype = "U" if evt.unavailability_type == "Unplanned" else "P"
        fuel = str(evt.fuel_type)[:15] if pd.notna(evt.fuel_type) else ""
        asset = str(evt.asset_name)[:15] if pd.notna(evt.asset_name) else ""

        label = f"[{utype}] {asset} ({fuel}, {mw})\n{reason}"
        label = textwrap.fill(label, width=40)

        # Alternate y offset
        if i % 2 == 0:
            y_text = y_max - (i // 2) * y_range * 0.08
        else:
            y_text = y_min + ((i - 1) // 2) * y_range * 0.08 + y_range * 0.05

        color = "#C0392B" if evt.unavailability_type == "Unplanned" else "#D68910"

        ax1.annotate(
            label,
            xy=(evt.event_start, price_at_event),
            xytext=(evt.event_start + pd.Timedelta(hours=6), y_text),
            fontsize=6.5,
            color=color,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color, alpha=0.6, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.85),
            zorder=10,
        )

        # Mark event start on price line
        ax1.plot(evt.event_start, price_at_event, "v" if evt.unavailability_type == "Unplanned" else "s",
                 color=color, markersize=6, zorder=9)

    ax1.set_ylabel("Day-ahead Price (EUR/MWh)", fontsize=11)
    ax1.legend(
        handles=[
            mpatches.Patch(color="#E74C3C", alpha=0.5, label="Unplanned outage"),
            mpatches.Patch(color="#F39C12", alpha=0.5, label="Planned outage"),
            mpatches.Patch(color="#1a1a2e", alpha=0.2, label=f"Night 21–06 ({ZONE_TZ.get(zone, 'UTC')})"),
        ],
        loc="upper left", fontsize=9,
    )

    # Bottom: stacked MW bars
    # Build hourly unavailable MW
    hours = pd.date_range(t_start, t_end, freq="h", tz="UTC")[:-1]
    hourly_mw = pd.DataFrame({"timestamp": hours, "unplanned_mw": 0.0, "planned_mw": 0.0})
    hourly_mw = hourly_mw.set_index("timestamp")

    for _, evt in events.iterrows():
        evt_end = min(evt.event_end, t_end) if pd.notna(evt.event_end) else t_end
        evt_hours = pd.date_range(evt.event_start.floor("h"), evt_end.floor("h"), freq="h", tz="UTC")
        mw = evt.unavailable_mw if pd.notna(evt.unavailable_mw) else 0
        col = "unplanned_mw" if evt.unavailability_type == "Unplanned" else "planned_mw"
        for h in evt_hours:
            if h in hourly_mw.index:
                hourly_mw.loc[h, col] += mw

    hourly_mw = hourly_mw.reset_index()
    ax2.fill_between(hourly_mw["timestamp"], 0, hourly_mw.unplanned_mw,
                     color="#E74C3C", alpha=0.6, label="Unplanned MW")
    ax2.fill_between(hourly_mw["timestamp"], hourly_mw.unplanned_mw,
                     hourly_mw.unplanned_mw + hourly_mw.planned_mw,
                     color="#F39C12", alpha=0.4, label="Planned MW")
    ax2.set_ylabel("Unavailable MW", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    period = f"{months[month_start-1]}–{months[month_end-1]} {year}"
    fig.suptitle(
        f"{zone} {period} — Day-ahead Prices with Outage Annotations\n{title_note}",
        fontsize=14, fontweight="bold",
    )

    plt.tight_layout()
    fname = f"{FIG_DIR}/entsoe_annotated_{zone}_{year}_{month_start:02d}_{month_end:02d}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close()


# ── Generate all showcase plots ───────────────────────────────────────────

for zone, year, m1, m2, note in SHOWCASE:
    make_annotated_plot(zone, year, m1, m2, note)
