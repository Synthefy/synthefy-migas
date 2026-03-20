"""
Nord Pool UMM Fetcher
=====================
Fetches REMIT Urgent Market Messages (UMMs) with free-text descriptions from
the Nord Pool public API (no API key needed) and joins them with ENTSO-E
day-ahead prices to produce annotated daily Parquet files.

Covers: FR, BE, FI, NO1, NL, DK1, SE1-4, EE, LT, LV, IE
(Germany/Spain/Poland are NOT on Nord Pool — they use EEX.)

Usage:
  uv run python scripts/fetch_nordpool_umm.py --zone FR --year 2024
  uv run python scripts/fetch_nordpool_umm.py --zone NO1 --year 2022
  uv run python scripts/fetch_nordpool_umm.py --zone FR --year 2024 --prices-csv data/entsoe_bulk/day_ahead_prices/FR.csv
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

# ── Zone mapping: short label → EIC code ──────────────────────────────────
ZONE_EIC = {
    "FR":  "10YFR-RTE------C",
    "BE":  "10YBE----------2",
    "NL":  "10YNL----------L",
    "FI":  "10YFI-1--------U",
    "NO1": "10YNO-1--------2",
    "NO2": "10YNO-2--------T",
    "NO4": "10YNO-4--------9",
    "NO5": "10Y1001A1001A48H",
    "DK1": "10YDK-1--------W",
    "DK2": "10YDK-2--------M",
    "SE1": "10Y1001A1001A44P",
    "SE2": "10Y1001A1001A45N",
    "SE3": "10Y1001A1001A46L",
    "SE4": "10Y1001A1001A47J",
    "EE":  "10Y1001A1001A39I",
    "LT":  "10YLT-1001A0008Q",
    "LV":  "10YLV-1001A00074",
    "IE":  "10Y1001A1001A59C",
}

# ── Fuel type codes (reverse-engineered from API data) ────────────────────
FUEL_TYPES = {
    1: "Biomass/CHP",
    4: "Natural Gas (CCGT)",
    5: "Coal/Hard coal",
    6: "Gas/Oil turbine",
    7: "Oil shale",
    8: "Peat/Biomass",
    10: "Pumped storage hydro",
    11: "Run-of-river hydro",
    12: "Reservoir hydro",
    14: "Nuclear",
    100: "Battery/Other",
}

BASE_URL = "https://ummapi.nordpoolgroup.com/messages"
PAGE_LIMIT = 1000  # API max is 2000, use 1000 for safety


def fetch_umm_messages(zone_eic: str, year: int, msg_types: list[str] | None = None) -> list[dict]:
    """Fetch all UMM messages for a zone and year, handling pagination."""
    if msg_types is None:
        msg_types = ["ProductionUnavailability"]

    start = f"{year}-01-01T00:00:00Z"
    end = f"{year + 1}-01-01T00:00:00Z"

    all_messages = []

    for msg_type in msg_types:
        offset = 0
        while True:
            params = {
                "areas": zone_eic,
                "eventStartDate": start,
                "eventStopDate": end,
                "messageTypes": msg_type,
                "limit": PAGE_LIMIT,
                "offset": offset,
            }

            print(f"  → Fetching {msg_type} offset={offset}...")
            resp = requests.get(BASE_URL, params=params, timeout=120)

            if resp.status_code == 413:
                print("  ✗ 413: too many records — try a smaller date range")
                break
            resp.raise_for_status()

            data = resp.json()
            items = data.get("items", [])
            total = data.get("total", 0)

            if not items:
                if offset == 0:
                    print(f"    0 messages for {msg_type}")
                break

            all_messages.extend(items)
            print(f"    ✓ {len(items)} messages (total available: {total})")

            offset += len(items)
            if offset >= total:
                break

            time.sleep(0.5)  # be polite

    print(f"  ✓ Total: {len(all_messages)} UMM messages")
    return all_messages


def parse_messages(messages: list[dict], zone_eic: str) -> pd.DataFrame:
    """Parse raw UMM JSON messages into a flat DataFrame."""
    rows = []

    for msg in messages:
        reason = msg.get("unavailabilityReason") or ""
        remarks = msg.get("remarks") or ""
        unavail_type = msg.get("unavailabilityType")  # 1=Unplanned, 2=Planned
        type_label = {1: "Unplanned", 2: "Planned"}.get(unavail_type, "Unknown")
        message_id = msg.get("messageId", "")
        version = msg.get("version", 1)
        publication_date = msg.get("publicationDate", "")

        # Extract production/generation units (two possible structures)
        units = msg.get("productionUnits") or []
        for gu in msg.get("generationUnits") or []:
            units.append(gu)

        if not units:
            # Message with no units — still record the text
            rows.append({
                "message_id": message_id,
                "version": version,
                "publication_date": publication_date,
                "unavailability_type": type_label,
                "reason_text": reason,
                "remarks": remarks,
                "asset_name": "",
                "fuel_type_code": None,
                "fuel_type": "",
                "area_name": "",
                "installed_capacity_mw": None,
                "event_start": None,
                "event_end": None,
                "unavailable_mw": None,
                "available_mw": None,
            })
            continue

        for unit in units:
            area = unit.get("areaName", "")
            area_eic = unit.get("areaEic", "")

            # Filter to only our target zone
            if zone_eic and area_eic != zone_eic:
                continue

            fuel_code = unit.get("fuelType")
            fuel_name = FUEL_TYPES.get(fuel_code, f"Unknown({fuel_code})")
            asset_name = unit.get("name") or unit.get("productionUnitName") or ""
            installed = unit.get("installedCapacity") or unit.get("productionUnitInstalledCapacity")

            for tp in unit.get("timePeriods", []):
                rows.append({
                    "message_id": message_id,
                    "version": version,
                    "publication_date": publication_date,
                    "unavailability_type": type_label,
                    "reason_text": reason,
                    "remarks": remarks,
                    "asset_name": asset_name,
                    "fuel_type_code": fuel_code,
                    "fuel_type": fuel_name,
                    "area_name": area,
                    "installed_capacity_mw": installed,
                    "event_start": tp.get("eventStart"),
                    "event_end": tp.get("eventStop"),
                    "unavailable_mw": tp.get("unavailableCapacity"),
                    "available_mw": tp.get("availableCapacity"),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["event_start"] = pd.to_datetime(df["event_start"], format="ISO8601", utc=True)
    df["event_end"] = pd.to_datetime(df["event_end"], format="ISO8601", utc=True)
    df["publication_date"] = pd.to_datetime(df["publication_date"], format="ISO8601", utc=True)
    df = df.sort_values("event_start").reset_index(drop=True)
    return df


def build_daily_text(umm_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Build daily text annotations from UMM messages.
    For each day, collect unique reason texts from active outages.
    """
    if umm_df.empty:
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        return pd.DataFrame({"t": dates.strftime("%Y-%m-%d"), "text": ""})

    # Expand each event to the days it covers within our year
    year_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    year_end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")

    daily_texts = {}  # date_str → set of text entries

    for _, row in umm_df.iterrows():
        evt_start = max(row["event_start"], year_start)
        evt_end = min(row["event_end"], year_end) if pd.notna(row["event_end"]) else year_end

        if evt_start >= evt_end:
            continue

        # Build a descriptive text entry
        parts = []
        if row["unavailability_type"] and row["unavailability_type"] != "Unknown":
            parts.append(row["unavailability_type"])
        if row["asset_name"]:
            cap_str = ""
            if row["unavailable_mw"]:
                cap_str = f", {int(row['unavailable_mw'])} MW unavailable"
            if row["fuel_type"]:
                parts.append(f"{row['asset_name']} ({row['fuel_type']}{cap_str})")
            else:
                parts.append(f"{row['asset_name']}{cap_str}")
        if row["reason_text"]:
            parts.append(row["reason_text"])
        if row["remarks"]:
            parts.append(row["remarks"])

        text_entry = ": ".join(parts[:2])
        if len(parts) > 2:
            text_entry += ". " + ". ".join(parts[2:])

        if not text_entry.strip():
            continue

        days = pd.date_range(evt_start.normalize(), evt_end.normalize(), freq="D")
        for day in days:
            ds = day.strftime("%Y-%m-%d")
            if ds not in daily_texts:
                daily_texts[ds] = set()
            daily_texts[ds].add(text_entry)

    # Build full year DataFrame
    all_dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    result = []
    for d in all_dates:
        ds = d.strftime("%Y-%m-%d")
        texts = daily_texts.get(ds, set())
        # Limit to 10 entries per day to keep it manageable
        combined = ". ".join(list(texts)[:10]) if texts else ""
        result.append({"t": ds, "text": combined})

    return pd.DataFrame(result)


def load_prices(prices_csv: str, year: int) -> pd.DataFrame:
    """Load ENTSO-E day-ahead prices and compute daily averages."""
    df = pd.read_csv(prices_csv, index_col=0, parse_dates=True)

    # Handle both single-column (value) and multi-column formats
    if "value" in df.columns:
        prices = df["value"]
    elif "price_eur_mwh" in df.columns:
        prices = df["price_eur_mwh"]
    else:
        # First column is the price
        prices = df.iloc[:, 0]

    prices = prices.to_frame(name="price")
    prices.index = pd.to_datetime(prices.index, utc=True)

    # Filter to target year
    prices = prices[prices.index.year == year]

    # Daily average
    daily = prices.resample("D").mean().reset_index()
    daily.columns = ["date", "y_t"]
    daily["t"] = daily["date"].dt.strftime("%Y-%m-%d")
    daily = daily[["t", "y_t"]].dropna()

    return daily


def main():
    parser = argparse.ArgumentParser(description="Fetch Nord Pool UMM messages with free text")
    parser.add_argument("--zone", required=True, help=f"Zone label: {', '.join(ZONE_EIC.keys())}")
    parser.add_argument("--year", required=True, type=int, help="Year to fetch (e.g. 2024)")
    parser.add_argument("--prices-csv", help="Path to ENTSO-E day-ahead prices CSV for this zone")
    parser.add_argument("--output-dir", default="data/entsoe_raw", help="Output directory")
    parser.add_argument("--include-transmission", action="store_true",
                        help="Also fetch TransmissionUnavailability messages")
    args = parser.parse_args()

    zone = args.zone.upper()
    if zone not in ZONE_EIC:
        print(f"Unknown zone '{zone}'. Available: {', '.join(ZONE_EIC.keys())}")
        return

    zone_eic = ZONE_EIC[zone]
    year = args.year
    out_dir = os.path.join(args.output_dir, f"{zone}_{year}")
    os.makedirs(out_dir, exist_ok=True)

    # ── Fetch UMMs ────────────────────────────────────────────────────────
    print(f"\n[1/3] Fetching Nord Pool UMMs for {zone} ({zone_eic}) — {year}")
    msg_types = ["ProductionUnavailability"]
    if args.include_transmission:
        msg_types.append("TransmissionUnavailability")

    raw_messages = fetch_umm_messages(zone_eic, year, msg_types)

    # Save raw JSON incrementally
    raw_path = os.path.join(out_dir, f"umm_raw_{zone}_{year}.json")
    with open(raw_path, "w") as f:
        json.dump(raw_messages, f, indent=2, default=str)
    print(f"  ✓ Saved raw JSON: {raw_path}")

    # ── Parse to DataFrame ────────────────────────────────────────────────
    print(f"\n[2/3] Parsing UMM messages...")
    umm_df = parse_messages(raw_messages, zone_eic)

    if umm_df.empty:
        print("  ✗ No messages parsed — check zone/year")
        return

    # Save parsed CSV
    umm_csv = os.path.join(out_dir, f"umm_parsed_{zone}_{year}.csv")
    umm_df.to_csv(umm_csv, index=False)
    print(f"  ✓ {len(umm_df)} parsed events")
    print(f"    With reason text: {(umm_df.reason_text != '').sum()}")
    print(f"    With remarks: {(umm_df.remarks != '').sum()}")
    print(f"    Unplanned: {(umm_df.unavailability_type == 'Unplanned').sum()}")
    print(f"    Planned: {(umm_df.unavailability_type == 'Planned').sum()}")
    print(f"  ✓ Saved: {umm_csv}")

    # ── Build daily annotated dataset ─────────────────────────────────────
    print(f"\n[3/3] Building daily annotations...")
    daily_text = build_daily_text(umm_df, year)

    if args.prices_csv:
        print(f"  Loading prices from {args.prices_csv}...")
        daily_prices = load_prices(args.prices_csv, year)
        result = daily_prices.merge(daily_text, on="t", how="left")
        result["text"] = result["text"].fillna("")
    else:
        print("  ⚠ No prices CSV provided — output will have text only (no y_t)")
        result = daily_text

    result = result.sort_values("t").reset_index(drop=True)

    # Save as Parquet
    parquet_dir = "data/entsoe"
    os.makedirs(parquet_dir, exist_ok=True)
    parquet_path = os.path.join(parquet_dir, f"{zone}_{year}.parquet")
    result.to_parquet(parquet_path, index=False)

    n_annotated = (result["text"] != "").sum()
    print(f"\n{'=' * 60}")
    print(f"DONE — {zone} {year}")
    print(f"{'=' * 60}")
    print(f"  Days total:     {len(result)}")
    print(f"  Days with text: {n_annotated} ({100 * n_annotated / len(result):.0f}%)")
    if "y_t" in result.columns:
        print(f"  Price range:    {result.y_t.min():.1f} — {result.y_t.max():.1f} EUR/MWh")
    print(f"\nFiles:")
    print(f"  {raw_path}")
    print(f"  {umm_csv}")
    print(f"  {parquet_path}")


if __name__ == "__main__":
    main()
