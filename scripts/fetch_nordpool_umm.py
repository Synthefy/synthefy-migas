"""
Nord Pool UMM Fetcher
=====================
Fetches REMIT Urgent Market Messages (UMMs) with free-text descriptions from
the Nord Pool public API (no API key needed).

Outputs per-zone/year Parquet + CSV files with daily text annotations:
  data/nordpool_umm/<ZONE>_<YEAR>.parquet

Columns:
  date            — date string YYYY-MM-DD
  text            — concatenated outage descriptions active that day

Covers: FR, BE, NL, FI, NO1, NO2, NO4, NO5, DK1, DK2, SE1–SE4, EE, LT, LV, IE
(Germany/Spain/Poland are NOT on Nord Pool — they use EEX.)

Usage:
  uv run python scripts/fetch_nordpool_umm.py
  uv run python scripts/fetch_nordpool_umm.py --zones FR DK_1 --start 2022 --end 2024
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ── Zone mapping: label → EIC code ───────────────────────────────────────────
ZONE_EIC = {
    "FR":   "10YFR-RTE------C",
    "BE":   "10YBE----------2",
    "NL":   "10YNL----------L",
    "FI":   "10YFI-1--------U",
    "NO_1": "10YNO-1--------2",
    "NO_2": "10YNO-2--------T",
    "NO_4": "10YNO-4--------9",
    "NO_5": "10Y1001A1001A48H",
    "DK_1": "10YDK-1--------W",
    "DK_2": "10YDK-2--------M",
    "SE_1": "10Y1001A1001A44P",
    "SE_2": "10Y1001A1001A45N",
    "SE_3": "10Y1001A1001A46L",
    "SE_4": "10Y1001A1001A47J",
    "EE":   "10Y1001A1001A39I",
    "LT":   "10YLT-1001A0008Q",
    "LV":   "10YLV-1001A00074",
    "IE":   "10Y1001A1001A59C",
}

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
PAGE_LIMIT = 1000


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_umm_messages(zone_eic: str, year: int) -> list[dict]:
    """Fetch all ProductionUnavailability UMMs for a zone and year."""
    start = f"{year}-01-01T00:00:00Z"
    end = f"{year + 1}-01-01T00:00:00Z"

    all_messages = []
    offset = 0

    while True:
        params = {
            "areas": zone_eic,
            "eventStartDate": start,
            "eventStopDate": end,
            "messageTypes": "ProductionUnavailability",
            "limit": PAGE_LIMIT,
            "offset": offset,
        }

        for attempt in range(3):
            try:
                resp = requests.get(BASE_URL, params=params, timeout=120)
                if resp.status_code == 413:
                    tqdm.write(f"    413: too many records")
                    return all_messages
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 2:
                    tqdm.write(f"    Failed: {e}")
                    return all_messages
                time.sleep(2 * (attempt + 1))

        data = resp.json()
        items = data.get("items", [])
        total = data.get("total", 0)

        if not items:
            break

        all_messages.extend(items)
        offset += len(items)
        if offset >= total:
            break

        time.sleep(0.3)

    return all_messages


def parse_messages(messages: list[dict], zone_eic: str) -> pd.DataFrame:
    """Parse raw UMM JSON into a flat DataFrame."""
    rows = []

    for msg in messages:
        reason = msg.get("unavailabilityReason") or ""
        remarks = msg.get("remarks") or ""
        unavail_type = {1: "Unplanned", 2: "Planned"}.get(
            msg.get("unavailabilityType"), "Unknown"
        )

        units = (msg.get("productionUnits") or []) + (msg.get("generationUnits") or [])

        if not units:
            rows.append({
                "unavailability_type": unavail_type,
                "reason_text": reason,
                "remarks": remarks,
                "asset_name": "",
                "fuel_type": "",
                "installed_capacity_mw": None,
                "event_start": None,
                "event_end": None,
                "unavailable_mw": None,
            })
            continue

        for unit in units:
            if zone_eic and unit.get("areaEic", "") != zone_eic:
                continue

            fuel_code = unit.get("fuelType")
            fuel_name = FUEL_TYPES.get(fuel_code, f"Unknown({fuel_code})")
            asset_name = unit.get("name") or unit.get("productionUnitName") or ""
            installed = unit.get("installedCapacity") or unit.get("productionUnitInstalledCapacity")

            for tp in unit.get("timePeriods", []):
                rows.append({
                    "unavailability_type": unavail_type,
                    "reason_text": reason,
                    "remarks": remarks,
                    "asset_name": asset_name,
                    "fuel_type": fuel_name,
                    "installed_capacity_mw": installed,
                    "event_start": tp.get("eventStart"),
                    "event_end": tp.get("eventStop"),
                    "unavailable_mw": tp.get("unavailableCapacity"),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["event_start"] = pd.to_datetime(df["event_start"], format="ISO8601", utc=True)
    df["event_end"] = pd.to_datetime(df["event_end"], format="ISO8601", utc=True)
    return df.sort_values("event_start").reset_index(drop=True)


def build_event_text(umm_df: pd.DataFrame) -> pd.DataFrame:
    """Build a text description for each event, keeping exact timestamps."""
    if umm_df.empty:
        return umm_df

    texts = []
    for _, row in umm_df.iterrows():
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
        texts.append(text_entry.strip())

    umm_df = umm_df.copy()
    umm_df["text"] = texts
    # Put event_start and event_end first
    cols = ["event_start", "event_end"] + [c for c in umm_df.columns if c not in ("event_start", "event_end")]
    return umm_df[cols]


# ── Main ──────────────────────────────────────────────────────────────────────

def fetch_zone_year(zone: str, zone_eic: str, year: int, out_dir: Path):
    """Fetch and save UMM data for a single zone/year."""
    out_parquet = out_dir / f"{zone}_{year}.parquet"
    out_csv = out_dir / f"{zone}_{year}.csv"

    raw = fetch_umm_messages(zone_eic, year)
    if not raw:
        tqdm.write(f"  {zone} {year}: no messages")
        return 0

    umm_df = parse_messages(raw, zone_eic)
    result = build_event_text(umm_df)

    result.to_parquet(out_parquet, index=False)
    result.to_csv(out_csv, index=False)

    tqdm.write(f"  {zone} {year}: {len(result)} events")
    return len(raw)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Nord Pool UMM text data for European bidding zones."
    )
    parser.add_argument("--zones", nargs="+", default=None,
                        help=f"Zones to fetch. Choices: {list(ZONE_EIC)}")
    parser.add_argument("--start", type=int, default=2020,
                        help="Start year (default: 2020)")
    parser.add_argument("--end", type=int, default=None,
                        help="End year inclusive (default: current year)")
    parser.add_argument("-o", "--output", default="data/nordpool_umm",
                        help="Output directory (default: data/nordpool_umm)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if output file exists")
    args = parser.parse_args()

    end_year = args.end or pd.Timestamp.now().year
    years = list(range(args.start, end_year + 1))

    zones = {z: eic for z, eic in ZONE_EIC.items()
             if args.zones is None or z in args.zones}
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for zone, zone_eic in zones.items():
        for year in years:
            out_path = out_dir / f"{zone}_{year}.parquet"
            if not args.force and out_path.exists():
                continue
            jobs.append((zone, zone_eic, year))

    if not jobs:
        print("All files exist. Use --force to re-download.")
        return

    pbar = tqdm(jobs, desc="Fetching UMMs", unit="zone-year")
    for zone, zone_eic, year in pbar:
        pbar.set_postfix_str(f"{zone} {year}")
        fetch_zone_year(zone, zone_eic, year, out_dir)

    pbar.close()
    print(f"\nDone → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
