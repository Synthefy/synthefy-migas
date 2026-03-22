"""
ENTSO-E Transparency Platform — Bulk REST API Download
=======================================================
Downloads all CSV data from the ENTSO-E Transparency Platform via HTTPS.
Works on any network — no SFTP/port-22 required.

Requires a Web API security token from your ENTSO-E account:
  https://transparency.entsoe.eu → My Account Settings → Generate Token

Usage:
    python entsoe_bulk_download.py --token YOUR_TOKEN
    python entsoe_bulk_download.py               # reads ENTSOE_TOKEN env var

Data lands in ./entsoe_data/<dataset>/<zone>.csv
"""

import os
import sys
import time
import argparse
from pathlib import Path

import pandas as pd

try:
    from entsoe import EntsoePandasClient
    from entsoe.exceptions import NoMatchingDataError, InvalidBusinessParameterError
except ImportError:
    sys.exit(
        "entsoe-py is required.\n"
        "Install it with:  uv add entsoe-py"
    )


# ── Bidding zones ────────────────────────────────────────────────────────────
ZONES = {
    "DE_LU": "10Y1001A1001A82H",
    "FR":    "10YFR-RTE------C",
    "GB":    "10YGB----------A",
    "ES":    "10YES-REE------0",
    "IT_N":  "10Y1001A1001A73I",
    "NL":    "10YNL----------L",
    "BE":    "10YBE----------2",
    "AT":    "10YAT-APG------L",
    "PL":    "10YPL-AREA-----S",
    "NO_1":  "10YNO-1--------2",
    "SE_1":  "10Y1001A1001A44P",
    "DK_1":  "10YDK-1--------W",
    "FI":    "10YFI-1--------U",
}

# ── Datasets ─────────────────────────────────────────────────────────────────
QUERIES = [
    ("day_ahead_prices",               "query_day_ahead_prices"),
    ("load_actual",                    "query_load"),
    ("load_forecast",                  "query_load_forecast"),
    ("generation_actual",              "query_generation"),
    ("generation_forecast_wind_solar", "query_wind_and_solar_forecast"),
    ("installed_capacity",             "query_installed_generation_capacity_per_unit"),
]

CHUNK_MONTHS = 3
RETRY_DELAY  = 2
MAX_RETRIES  = 3


def date_chunks(start: pd.Timestamp, end: pd.Timestamp, months: int):
    current = start
    while current < end:
        next_ = current + pd.DateOffset(months=months)
        yield current, min(next_, end)
        current = next_


def download_series(client, method, zone_code, start, end):
    fn = getattr(client, method)
    frames = []
    for chunk_start, chunk_end in date_chunks(start, end, CHUNK_MONTHS):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = fn(zone_code, start=chunk_start, end=chunk_end)
                if result is not None:
                    frames.append(result)
                break
            except (NoMatchingDataError, InvalidBusinessParameterError):
                break
            except Exception as e:
                if attempt == MAX_RETRIES:
                    print(f"    !! {e}")
                else:
                    time.sleep(RETRY_DELAY * attempt)
    if not frames:
        return None
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined.sort_index()


def save(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, pd.DataFrame):
        data.to_csv(path)
    else:
        data.to_frame(name="value").to_csv(path)


def main():
    parser = argparse.ArgumentParser(
        description="Bulk-download ENTSO-E data via HTTPS REST API."
    )
    parser.add_argument("--token",    default=None, help="ENTSO-E Web API security token.")
    parser.add_argument("--start",    default="2020-01-01", help="Start date YYYY-MM-DD.")
    parser.add_argument("--end",      default=None, help="End date YYYY-MM-DD (default: today).")
    parser.add_argument("-o", "--output", default="./entsoe_data", help="Output directory.")
    parser.add_argument("--zones",    nargs="+", default=None, help=f"Zones to download. Choices: {list(ZONES)}")
    parser.add_argument("--datasets", nargs="+", default=None, help=f"Datasets. Choices: {[q[0] for q in QUERIES]}")
    parser.add_argument("--force",    action="store_true", help="Re-download existing files.")
    args = parser.parse_args()

    token = args.token or os.environ.get("ENTSOE_TOKEN") or input("ENTSO-E security token: ").strip()
    start = pd.Timestamp(args.start, tz="UTC")
    end   = pd.Timestamp(args.end or pd.Timestamp.now().strftime("%Y-%m-%d"), tz="UTC")

    zones   = {k: v for k, v in ZONES.items() if not args.zones    or k in args.zones}
    queries = [(n, m) for n, m in QUERIES       if not args.datasets or n in args.datasets]
    out_root = Path(args.output)

    client = EntsoePandasClient(api_key=token)
    total_jobs = len(queries) * len(zones)
    print(f"Connected. {len(queries)} dataset(s) × {len(zones)} zone(s) = {total_jobs} total  [{start.date()} → {end.date()}]\n")

    completed = 0
    downloaded = 0
    t0 = time.time()

    for ds_name, method in queries:
        print(f"── {ds_name}")
        for zone_name, zone_code in zones.items():
            completed += 1
            pct = completed / total_jobs * 100
            elapsed = time.time() - t0
            eta = (elapsed / completed) * (total_jobs - completed) if completed > 1 else 0
            progress = f"[{completed}/{total_jobs}  {pct:.0f}%  eta {eta/60:.1f}m]"

            out_path = out_root / ds_name / f"{zone_name}.csv"
            if not args.force and out_path.exists():
                print(f"   {zone_name:8s}  skip (exists)  {progress}")
                continue
            data = download_series(client, method, zone_code, start, end)
            if data is None or (hasattr(data, "empty") and data.empty):
                print(f"   {zone_name:8s}  no data  {progress}")
            else:
                save(data, out_path)
                downloaded += 1
                print(f"   {zone_name:8s}  {len(data)} rows saved  {progress}")
            time.sleep(0.5)

    elapsed_total = time.time() - t0
    print(f"\nDone. {downloaded} file(s) downloaded in {elapsed_total/60:.1f}m → {out_root.resolve()}")


if __name__ == "__main__":
    main()
