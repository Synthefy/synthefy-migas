"""
ENTSO-E Full Data Fetcher
=========================
Downloads day-ahead prices, actual load, actual generation by fuel type,
and installed generation capacity for European bidding zones via entsoe-py.

Outputs one Parquet file per zone/year at native resolution (15-min or hourly):
  data/entsoe/<ZONE>_<YEAR>.parquet

Columns:
  timestamp                   — datetime with timezone
  price                       — day-ahead price (EUR/MWh)
  load                        — actual load (MW)
  generation/<fuel>            — generation per fuel type (MW)
  capacity/<fuel>              — installed capacity per fuel type (MW, forward-filled)

Prerequisites:
  uv add entsoe-py pandas pyarrow
  export ENTSOE_TOKEN='your-token'

Usage:
  uv run python scripts/fetch_entsoe_all.py
  uv run python scripts/fetch_entsoe_all.py --zones DE_LU FR --start 2022-01-01 --end 2023-01-01
"""

import os
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from entsoe import EntsoePandasClient
    from entsoe.exceptions import NoMatchingDataError, InvalidBusinessParameterError
except ImportError:
    sys.exit("entsoe-py is required.\nInstall with:  uv add entsoe-py")

# ── Bidding zones ─────────────────────────────────────────────────────────────
# entsoe-py accepts these labels directly (more reliable than EIC codes)
ZONES = [
    "DE_LU", "FR", "ES", "PL",
    "NO_1", "NO_2", "NL", "BE", "AT",
    "FI", "GB", "SE_1", "SE_3",
    "DK_1", "DK_2", "IT_NORD",
]

CHUNK_MONTHS = 3
MAX_RETRIES = 3
RETRY_DELAY = 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def date_chunks(start: pd.Timestamp, end: pd.Timestamp, months: int):
    """Yield (chunk_start, chunk_end) pairs of at most `months` length."""
    current = start
    while current < end:
        next_ = min(current + pd.DateOffset(months=months), end)
        yield current, next_
        current = next_


def fetch_chunked(client, method_name: str, zone_code: str,
                  start: pd.Timestamp, end: pd.Timestamp, **kwargs):
    """Call an entsoe-py query in chunks, concatenate results."""
    fn = getattr(client, method_name)
    frames = []
    for cs, ce in date_chunks(start, end, CHUNK_MONTHS):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = fn(zone_code, start=cs, end=ce, **kwargs)
                if result is not None:
                    frames.append(result)
                break
            except (NoMatchingDataError, InvalidBusinessParameterError):
                break
            except Exception as e:
                if attempt == MAX_RETRIES:
                    print(f"    !! {method_name} chunk {cs.date()}→{ce.date()}: {e}")
                else:
                    time.sleep(RETRY_DELAY * attempt)
        time.sleep(0.5)

    if not frames:
        return None
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined.sort_index()


def slugify(col_name) -> str:
    """Turn 'Fossil Brown coal/Lignite' into 'fossil_brown_coal_lignite'.
    Also handles tuple columns from MultiIndex."""
    if isinstance(col_name, tuple):
        col_name = col_name[0]
    return (
        str(col_name).lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
        .strip("_")
    )


# ── Fetch functions ───────────────────────────────────────────────────────────

def fetch_prices(client, zone_code, start, end) -> pd.DataFrame | None:
    """Day-ahead prices → DataFrame with DatetimeIndex and 'price_eur_mwh' col."""
    print("  Prices...", end=" ", flush=True)
    series = fetch_chunked(client, "query_day_ahead_prices", zone_code, start, end)
    if series is None:
        print("no data")
        return None
    df = series.to_frame(name="price")
    print(f"{len(df)} rows")
    return df


def fetch_load(client, zone_code, start, end) -> pd.DataFrame | None:
    """Actual load → DataFrame with DatetimeIndex and 'load_mw' col."""
    print("  Load...", end=" ", flush=True)
    df = fetch_chunked(client, "query_load", zone_code, start, end)
    if df is None:
        print("no data")
        return None
    # query_load returns DataFrame with 'Actual Load' column
    df = df.rename(columns={"Actual Load": "load"})
    if "load" not in df.columns:
        df.columns = ["load"]
    print(f"{len(df)} rows")
    return df[["load"]]


def fetch_generation(client, zone_code, start, end) -> pd.DataFrame | None:
    """Actual generation by fuel type → DataFrame with generation/<fuel> columns."""
    print("  Generation...", end=" ", flush=True)
    df = fetch_chunked(client, "query_generation", zone_code, start, end)
    if df is None:
        print("no data")
        return None
    # Filter to 'Actual Aggregated' — columns may be MultiIndex, tuples, or strings
    agg_cols = [col for col in df.columns
                if isinstance(col, tuple) and col[1] == "Actual Aggregated"]
    if agg_cols:
        df = df[agg_cols]
    df.columns = [f"generation/{slugify(c)}" for c in df.columns]
    print(f"{len(df)} rows, {len(df.columns)} fuel types")
    return df


def fetch_capacity(client, zone_code, start, end) -> pd.DataFrame | None:
    """Installed generation capacity per fuel type → single-row or annual."""
    print("  Capacity...", end=" ", flush=True)
    df = fetch_chunked(client, "query_installed_generation_capacity",
                       zone_code, start, end)
    if df is None:
        print("no data")
        return None
    df.columns = [f"capacity/{slugify(c)}" for c in df.columns]
    print(f"{len(df)} rows, {len(df.columns)} fuel types")
    return df


# ── Daily aggregation & join ──────────────────────────────────────────────────

def join_raw(prices_df, load_df, gen_df, cap_df) -> pd.DataFrame:
    """Join all datasets at their native resolution (15-min or hourly)."""

    frames = {}

    if prices_df is not None:
        prices_df.columns = ["price"]
        frames["prices"] = prices_df

    if load_df is not None:
        frames["load"] = load_df

    if gen_df is not None:
        frames["generation"] = gen_df

    if not frames:
        return pd.DataFrame()

    # Join on datetime index — outer join keeps all timestamps
    result = list(frames.values())[0]
    for df in list(frames.values())[1:]:
        result = result.join(df, how="outer")

    # Capacity is annual/sparse — forward-fill to every row
    if cap_df is not None and not cap_df.empty:
        cap_filled = cap_df.reindex(result.index, method="ffill")
        result = result.join(cap_filled, how="left")

    # Keep datetime as a column called "timestamp"
    result = result.reset_index()
    result = result.rename(columns={result.columns[0]: "timestamp"})
    result = result.sort_values("timestamp").reset_index(drop=True)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch ENTSO-E data (prices, load, generation, capacity) "
                    "and save Parquet files at native resolution."
    )
    parser.add_argument("--token", default=None,
                        help="ENTSO-E API token (or set ENTSOE_TOKEN env var)")
    parser.add_argument("--start", default="2020-01-01",
                        help="Start date YYYY-MM-DD (default: 2020-01-01)")
    parser.add_argument("--end", default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--zones", nargs="+", default=None,
                        help=f"Zones to fetch. Choices: {list(ZONES)}")
    parser.add_argument("-o", "--output", default="data/entsoe",
                        help="Output directory (default: data/entsoe)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if output file exists")
    args = parser.parse_args()

    token = (args.token
             or os.environ.get("ENTSOE_TOKEN")
             or os.environ.get("ENTSOE_API_TOKEN"))
    if not token:
        print("No API token found. Set ENTSOE_TOKEN env var or pass --token.")
        sys.exit(1)

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end or pd.Timestamp.now().strftime("%Y-%m-%d"),
                       tz="UTC")

    zones = [z for z in ZONES if args.zones is None or z in args.zones]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = EntsoePandasClient(api_key=token)

    # Build list of (zone, zone_code, year) jobs
    years = list(range(start.year, end.year + 1))
    jobs = []
    for zone in zones:
        for year in years:
            year_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
            year_end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
            year_start = max(year_start, start)
            year_end = min(year_end, end)
            if year_start < year_end:
                jobs.append((zone, year, year_start, year_end))

    pbar = tqdm(jobs, desc="Fetching", unit="zone-year")
    for zone, year, year_start, year_end in pbar:
        pbar.set_postfix_str(f"{zone} {year}")

        out_path = out_dir / f"{zone}_{year}.parquet"
        if not args.force and out_path.exists():
            continue

        with ThreadPoolExecutor(max_workers=4) as pool:
            f_prices = pool.submit(fetch_prices, client, zone, year_start, year_end)
            f_load   = pool.submit(fetch_load, client, zone, year_start, year_end)
            f_gen    = pool.submit(fetch_generation, client, zone, year_start, year_end)
            f_cap    = pool.submit(fetch_capacity, client, zone, year_start, year_end)

        prices = f_prices.result()
        load   = f_load.result()
        gen    = f_gen.result()
        cap    = f_cap.result()

        daily = join_raw(prices, load, gen, cap)
        if daily.empty:
            tqdm.write(f"  No data for {zone} {year}")
            continue

        daily.to_parquet(out_path, index=False)
        daily.to_csv(out_path.with_suffix(".csv"), index=False)
        tqdm.write(f"  {out_path}  ({len(daily)} days)")

    pbar.close()
    print(f"\nDone → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
