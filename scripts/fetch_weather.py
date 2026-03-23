"""Standalone script to fetch historical weather data from Open-Meteo for Nordic/European zones.

Usage:
    python scripts/fetch_weather_data.py --start 2015-01-01 --end 2025-12-31 --mode minutely_15
    python scripts/fetch_weather_data.py --start 2020-01-01 --end 2025-06-30 --mode hourly
    python scripts/fetch_weather_data.py --mode daily  # defaults to 2015-01-01 - 2025-12-31

Requires OPENMETEO_API_KEY env var (or pass --free for free tier with rate limits).
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

LOCATIONS = [
    {
        "zone": "DK_1",
        "region": "West Denmark",
        "city": "Aarhus",
        "lat": 56.16,
        "lon": 10.20,
    },
    {
        "zone": "DK_2",
        "region": "East Denmark",
        "city": "Copenhagen",
        "lat": 55.68,
        "lon": 12.57,
    },
    {"zone": "FI", "region": "Finland", "city": "Helsinki", "lat": 60.17, "lon": 24.94},
    {"zone": "FR", "region": "France", "city": "Paris", "lat": 48.86, "lon": 2.35},
    {
        "zone": "NO_1",
        "region": "South Norway",
        "city": "Oslo",
        "lat": 59.91,
        "lon": 10.75,
    },
    {
        "zone": "NO_2",
        "region": "Southwest Norway",
        "city": "Kristiansand",
        "lat": 58.15,
        "lon": 8.00,
    },
    {
        "zone": "SE_1",
        "region": "North Sweden",
        "city": "Lulea",
        "lat": 65.58,
        "lon": 22.15,
    },
    {
        "zone": "SE_3",
        "region": "South Sweden",
        "city": "Stockholm",
        "lat": 59.33,
        "lon": 18.07,
    },
]

# Open-Meteo variable mapping per mode
MODE_CONFIG = {
    "minutely_15": {
        "param_key": "minutely_15",
        "variables": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "precipitation",
        ],
        "rename": {
            "time": "timestamp",
            "temperature_2m": "temperature_celsius",
            "relative_humidity_2m": "relative_humidity_percent",
            "wind_speed_10m": "wind_speed_kmh",
            "precipitation": "precipitation_mm",
        },
        "chunk_days": 90,  # 15-min data is large; chunk by ~3 months
    },
    "hourly": {
        "param_key": "hourly",
        "variables": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "cloud_cover",
            "precipitation",
        ],
        "rename": {
            "time": "timestamp",
            "temperature_2m": "temperature_celsius",
            "relative_humidity_2m": "relative_humidity_percent",
            "wind_speed_10m": "wind_speed_kmh",
            "cloud_cover": "cloud_cover_percent",
            "precipitation": "precipitation_mm",
        },
        "chunk_days": 365,
    },
    "daily": {
        "param_key": "daily",
        "variables": [
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "relative_humidity_2m_mean",
            "wind_speed_10m_max",
            "cloud_cover_mean",
            "precipitation_sum",
        ],
        "rename": {
            "time": "timestamp",
            "temperature_2m_mean": "temperature_celsius",
            "temperature_2m_max": "temperature_max_celsius",
            "temperature_2m_min": "temperature_min_celsius",
            "relative_humidity_2m_mean": "relative_humidity_percent",
            "wind_speed_10m_max": "wind_speed_kmh",
            "cloud_cover_mean": "cloud_cover_percent",
            "precipitation_sum": "precipitation_mm",
        },
        "chunk_days": 3650,  # daily is small, can do large chunks
    },
}


def fetch_chunk(
    lat: float, lon: float, start: str, end: str, mode: str, api_key: str | None
) -> pd.DataFrame:
    """Fetch a single chunk of weather data from Open-Meteo archive API."""
    cfg = MODE_CONFIG[mode]

    params: dict = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "timezone": "auto",
        cfg["param_key"]: ",".join(cfg["variables"]),
    }

    if api_key not in (None, "free_tier"):
        params["api_key"] = api_key
        url = "https://customer-api.open-meteo.com/v1/archive"
    else:
        url = "https://archive-api.open-meteo.com/v1/archive"

    for attempt in range(5):
        try:
            resp = requests.get(url, params=params, timeout=60)
            data = resp.json()

            if data.get("error"):
                reason = data.get("reason", "Unknown error")
                if "Too Many Requests" in reason or resp.status_code == 429:
                    wait = 2**attempt * 5
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"API error: {reason}")

            results = data[cfg["param_key"]]
            df = pd.DataFrame(results)
            df.rename(columns=cfg["rename"], inplace=True)
            return df

        except (requests.ConnectionError, requests.Timeout) as e:
            wait = 2**attempt * 2
            print(f"    Connection error ({e}), retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"Failed after 5 attempts for {start} to {end}")


def fetch_location(
    loc: dict,
    start_date: str,
    end_date: str,
    mode: str,
    api_key: str | None,
    output_dir: Path,
) -> Path:
    """Fetch full date range for one location, chunked to respect API limits."""
    cfg = MODE_CONFIG[mode]
    chunk_days = cfg["chunk_days"]

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    chunks: list[pd.DataFrame] = []
    current = start

    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        s_str = current.strftime("%Y-%m-%d")
        e_str = chunk_end.strftime("%Y-%m-%d")
        print(f"  {loc['zone']}: {s_str} -> {e_str}")

        df = fetch_chunk(loc["lat"], loc["lon"], s_str, e_str, mode, api_key)
        chunks.append(df)
        current = chunk_end + timedelta(days=1)

        # Small delay between chunks to be nice to the API
        time.sleep(0.5)

    full_df = pd.concat(chunks, ignore_index=True)
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])
    full_df.sort_values("timestamp", inplace=True)
    full_df.reset_index(drop=True, inplace=True)

    filename = f"weather_{loc['zone']}_{loc['city'].lower()}_{mode}_{start_date}_{end_date}.parquet"
    out_path = output_dir / filename
    full_df.to_parquet(out_path, index=False)

    print(f"  -> Saved {len(full_df)} rows to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Open-Meteo historical weather data for Nordic/EU zones"
    )
    parser.add_argument(
        "--start",
        default="2015-01-01",
        help="Start date YYYY-MM-DD (default: 2015-01-01)",
    )
    parser.add_argument(
        "--end", default="2025-12-31", help="End date YYYY-MM-DD (default: 2025-12-31)"
    )
    parser.add_argument(
        "--mode",
        default="minutely_15",
        choices=MODE_CONFIG.keys(),
        help="Temporal resolution (default: minutely_15)",
    )
    parser.add_argument(
        "--output-dir",
        default="weather_data",
        help="Output directory (default: weather_data)",
    )
    parser.add_argument(
        "--free",
        action="store_true",
        help="Use free tier (no API key, stricter rate limits)",
    )
    parser.add_argument(
        "--zones", nargs="*", help="Only fetch specific zones, e.g. --zones DK_1 FI FR"
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENMETEO_API_KEY") if not args.free else "free_tier"
    if not api_key:
        print(
            "Error: OPENMETEO_API_KEY not set. Use --free for free tier or set the env var.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    locations = LOCATIONS
    if args.zones:
        locations = [loc for loc in LOCATIONS if loc["zone"] in args.zones]
        if not locations:
            print(f"Error: No matching zones found for {args.zones}", file=sys.stderr)
            sys.exit(1)

    print(f"Fetching {args.mode} weather data from {args.start} to {args.end}")
    print(f"Locations: {[loc['zone'] for loc in locations]}")
    print(f"Output: {output_dir}/")
    print()

    for loc in locations:
        print(f"[{loc['zone']}] {loc['city']} ({loc['lat']}, {loc['lon']})")
        try:
            fetch_location(loc, args.start, args.end, args.mode, api_key, output_dir)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
