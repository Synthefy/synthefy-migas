"""
Weekly annotated time series — 7-day chunks showing ALL outages.

Usage:
  uv run python scripts/entsoe_weekly_annotated.py
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

FIG_DIR = os.path.join(_ROOT_DIR, "figures/weekly")
os.makedirs(FIG_DIR, exist_ok=True)
DATA_DIR = os.path.join(_ROOT_DIR, "data/entsoe_examples_new")

# ── Zone → local timezone mapping ────────────────────────────────────────

ZONE_TZ = {
    "NO1": "Europe/Oslo",      # CET/CEST (UTC+1/+2)
    "NO2": "Europe/Oslo",
    "SE1": "Europe/Stockholm", # CET/CEST
    "SE3": "Europe/Stockholm",
    "DK1": "Europe/Copenhagen",# CET/CEST
    "DK2": "Europe/Copenhagen",
    "FI":  "Europe/Helsinki",  # EET/EEST (UTC+2/+3)
}

NIGHT_START_LOCAL = 21  # 9 PM local
NIGHT_END_LOCAL = 6     # 6 AM local


def shade_night(ax, t_start, t_end, zone):
    """Shade nighttime hours (21:00-06:00 local) on a UTC-indexed axis."""
    import pytz
    tz = pytz.timezone(ZONE_TZ.get(zone, "Europe/Oslo"))

    current = t_start.normalize()  # midnight UTC
    while current < t_end + pd.Timedelta(days=1):
        # Night block: local 21:00 → local 06:00 next day
        night_start_local = tz.localize(
            pd.Timestamp(current.year, current.month, current.day, NIGHT_START_LOCAL)
        )
        night_end_local = night_start_local + pd.Timedelta(hours=9)  # 21:00 → 06:00

        ns_utc = night_start_local.astimezone(pytz.UTC)
        ne_utc = night_end_local.astimezone(pytz.UTC)

        # Clip to plot range
        ns_clipped = max(ns_utc, t_start)
        ne_clipped = min(ne_utc, t_end)

        if ns_clipped < ne_clipped:
            ax.axvspan(ns_clipped, ne_clipped, alpha=0.06, color="#1a1a2e", zorder=0)

        current += pd.Timedelta(days=1)


# ── Config: which zone/year/period to plot ────────────────────────────────

ZONE = "FI"
YEAR = 2022
# Generate weekly chunks for Nov-Dec
START_DATE = "2022-11-01"
END_DATE = "2023-01-01"


def make_weekly_plot(zone, year, week_start, week_end, prices, events, week_num):
    """One 7-day plot showing ALL outages."""

    p = prices[(prices.timestamp_utc >= week_start) & (prices.timestamp_utc < week_end)].copy()
    wk_events = events[(events.event_start >= week_start) & (events.event_start < week_end)].copy()
    wk_events = wk_events.sort_values("event_start")

    if p.empty:
        return

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(18, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.05},
    )

    # ── Top: price line ───────────────────────────────────────────────────

    # Shade nighttime hours first (behind everything)
    shade_night(ax1, week_start, week_end, zone)
    shade_night(ax2, week_start, week_end, zone)

    ax1.plot(p.timestamp_utc, p.price_eur_mwh, lw=1.2, color="#2C3E50", zorder=3)
    ax1.fill_between(p.timestamp_utc, 0, p.price_eur_mwh, alpha=0.05, color="#2C3E50")

    # Shade each outage period
    for _, evt in wk_events.iterrows():
        evt_end_ts = min(evt.event_end, week_end) if pd.notna(evt.event_end) else week_end
        color = "#E74C3C" if evt.unavailability_type == "Unplanned" else "#F39C12"
        ax1.axvspan(evt.event_start, evt_end_ts, alpha=0.06, color=color, linewidth=0)

    # ── Annotate ALL events ───────────────────────────────────────────────

    y_max = p.price_eur_mwh.max()
    y_min = p.price_eur_mwh.min()
    y_range = max(y_max - y_min, 20)

    # Group events that are close in time to stagger vertically
    n_events = len(wk_events)
    used_positions = []

    for i, (_, evt) in enumerate(wk_events.iterrows()):
        # Find price at event time
        closest_idx = (p.timestamp_utc - evt.event_start).abs().idxmin()
        price_at_event = p.loc[closest_idx, "price_eur_mwh"]

        # Build label
        mw = f"{evt.unavailable_mw:.0f}MW" if pd.notna(evt.unavailable_mw) else ""
        reason = str(evt.reason_text)[:45] if pd.notna(evt.reason_text) else ""
        utype = "U" if evt.unavailability_type == "Unplanned" else "P"
        fuel = str(evt.fuel_type)[:12] if pd.notna(evt.fuel_type) else ""
        asset = str(evt.asset_name)[:12] if pd.notna(evt.asset_name) else ""

        label = f"[{utype}] {asset} ({fuel}, {mw})\n{reason}"

        color = "#C0392B" if evt.unavailability_type == "Unplanned" else "#D68910"

        # Stagger: alternate between top and bottom, spread evenly
        if n_events <= 10:
            if i % 2 == 0:
                y_text = y_max + y_range * 0.15 + (i // 2) * y_range * 0.12
            else:
                y_text = y_min - y_range * 0.15 - ((i - 1) // 2) * y_range * 0.12
        else:
            # More events: use tighter spacing
            slot = i / max(n_events - 1, 1)
            y_text = y_min - y_range * 0.1 + slot * (y_range * 1.6 + y_range * 0.3)

        # Offset x slightly to reduce arrow overlap
        x_offset = pd.Timedelta(hours=2 + (i % 3) * 2)

        ax1.annotate(
            label,
            xy=(evt.event_start, price_at_event),
            xytext=(evt.event_start + x_offset, y_text),
            fontsize=6,
            color=color,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color, alpha=0.5, lw=0.7,
                            connectionstyle="arc3,rad=0.1"),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=color, alpha=0.9, linewidth=0.8),
            zorder=10,
        )

        # Marker on price line
        marker = "v" if evt.unavailability_type == "Unplanned" else "^"
        ax1.plot(evt.event_start, price_at_event, marker,
                 color=color, markersize=7, zorder=9)

    ax1.set_ylabel("Day-ahead Price (EUR/MWh)", fontsize=11)
    tz_name = ZONE_TZ.get(zone, "UTC")
    ax1.legend(
        handles=[
            mpatches.Patch(color="#E74C3C", alpha=0.6, label=f"Unplanned ({len(wk_events[wk_events.unavailability_type=='Unplanned'])})"),
            mpatches.Patch(color="#F39C12", alpha=0.6, label=f"Planned ({len(wk_events[wk_events.unavailability_type=='Planned'])})"),
            mpatches.Patch(color="#1a1a2e", alpha=0.2, label=f"Night 21:00–06:00 ({tz_name})"),
        ],
        loc="upper right", fontsize=9,
    )

    # ── Bottom: MW unavailable ────────────────────────────────────────────

    hours = pd.date_range(week_start, week_end, freq="h", tz="UTC")[:-1]
    hourly_mw = pd.DataFrame({"timestamp": hours, "unplanned_mw": 0.0, "planned_mw": 0.0})
    hourly_mw = hourly_mw.set_index("timestamp")

    # Include events that started before this week but are still active
    all_active = events[
        (events.event_start < week_end) &
        (events.event_end > week_start)
    ]

    for _, evt in all_active.iterrows():
        s = max(evt.event_start.floor("h"), week_start)
        e = min(evt.event_end.ceil("h"), week_end) if pd.notna(evt.event_end) else week_end
        evt_hours = pd.date_range(s, e, freq="h", tz="UTC")
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
    ax2.legend(loc="upper right", fontsize=9)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%a %b %d\n%H:%M"))
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    ax2.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))

    ws = week_start.strftime("%b %d")
    we = (week_end - pd.Timedelta(days=1)).strftime("%b %d, %Y")
    fig.suptitle(
        f"{zone} — {ws} to {we} (Week {week_num})\n"
        f"{len(wk_events)} outage events ({len(wk_events[wk_events.unavailability_type=='Unplanned'])} unplanned)",
        fontsize=14, fontweight="bold",
    )

    fname = f"{FIG_DIR}/week_{zone}_{year}_{week_num:02d}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname} ({len(wk_events)} events)")
    plt.close()


# ── Load data ─────────────────────────────────────────────────────────────

prices = pd.read_csv(f"{DATA_DIR}/{ZONE}_{YEAR}/entsoe_prices_{ZONE}.csv")
prices["timestamp_utc"] = pd.to_datetime(prices["timestamp_utc"], utc=True)
prices = prices.sort_values("timestamp_utc")

umm = pd.read_csv(f"{DATA_DIR}/{ZONE}_{YEAR}/umm_parsed_{ZONE}_{YEAR}.csv")
umm["event_start"] = pd.to_datetime(umm["event_start"], format="ISO8601", utc=True)
umm["event_end"] = pd.to_datetime(umm["event_end"], format="ISO8601", utc=True)

print(f"Loaded {ZONE} {YEAR}: {len(prices)} hourly prices, {len(umm)} events")

# ── Generate weekly plots ─────────────────────────────────────────────────

start = pd.Timestamp(START_DATE, tz="UTC")
end = pd.Timestamp(END_DATE, tz="UTC")

week_num = 1
current = start
while current < end:
    week_end = current + pd.Timedelta(days=7)
    make_weekly_plot(ZONE, YEAR, current, week_end, prices, umm, week_num)
    current = week_end
    week_num += 1

print(f"\nAll weekly plots saved to {FIG_DIR}/")
