"""
Full pipeline: fetch ENTSO-E prices + Nord Pool UMM text → training Parquet
=============================================================================
Run once, forget:

  ENTSOE_API_TOKEN=a447c244-ce1a-4568-bd84-cb601805204a uv run python scripts/build_training_data.py

Produces:
  data/entsoe/*.parquet          — one file per zone/year
  data/entsoe/entsoe_training.parquet — combined training dataset
"""

import glob
import os
import subprocess
import sys
import time

from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────

# Zones with good Nord Pool UMM text + ENTSO-E price availability
# Format: (nordpool_label, entsoe_eic, supply_mix)
ZONES = [
    ("NO1", "10YNO-1--------2",  "Hydro"),
    ("NO2", "10YNO-2--------T",  "Hydro"),
    ("FI",  "10YFI-1--------U",  "Nuclear/CHP"),
    ("DK1", "10YDK-1--------W",  "Wind/Gas"),
    ("DK2", "10YDK-2--------M",  "Wind/Thermal"),
    ("SE1", "10Y1001A1001A44P",  "Hydro/Wind"),
    ("SE3", "10Y1001A1001A46L",  "Nuclear/Mixed"),
]

# 2016–2025 = 10 years. Nord Pool UMM data is available from ~2013 but thin
# before 2015. ENTSO-E prices reliable from 2015+.
# 7 zones × 10 years = ~25,500 daily rows.
YEARS = list(range(2016, 2026))

# Existing 2024 bulk prices (already in repo, no fetch needed)
BULK_PRICES_2024 = {
    "NO1": "data/entsoe_bulk/day_ahead_prices/NO_1.csv",
    "FI":  "data/entsoe_bulk/day_ahead_prices/FI.csv",
    "DK1": "data/entsoe_bulk/day_ahead_prices/DK_1.csv",
    "SE1": "data/entsoe_bulk/day_ahead_prices/SE_1.csv",
}

ENTSOE_TOKEN = os.environ.get("ENTSOE_API_TOKEN", "")
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)


def run(cmd: list[str], description: str) -> bool:
    """Run a subprocess, return True on success."""
    print(f"\n{'─' * 60}")
    print(f"  {description}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'─' * 60}")
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    if result.returncode != 0:
        print(f"  ✗ FAILED: {description}")
        return False
    return True


def prices_path(zone: str, year: int) -> str:
    return os.path.join(ROOT_DIR, f"data/entsoe_raw/{zone}_{year}/entsoe_prices_{zone}.csv")


def parquet_path(zone: str, year: int) -> str:
    return os.path.join(ROOT_DIR, f"data/entsoe/{zone}_{year}.parquet")


def main():
    if not ENTSOE_TOKEN:
        print("ERROR: Set ENTSOE_API_TOKEN environment variable")
        sys.exit(1)

    total_combos = len(ZONES) * len(YEARS)
    done = 0
    skipped = 0
    failed = []

    print(f"=" * 60)
    print(f"Building training data: {len(ZONES)} zones × {len(YEARS)} years = {total_combos} combos")
    print(f"Zones: {', '.join(z[0] for z in ZONES)}")
    print(f"Years: {YEARS[0]}–{YEARS[-1]}")
    print(f"=" * 60)

    for zone_label, zone_eic, supply_mix in ZONES:
        for year in YEARS:
            combo = f"{zone_label}_{year}"
            done += 1
            pq = parquet_path(zone_label, year)

            # Skip if already done
            if os.path.exists(pq):
                print(f"\n[{done}/{total_combos}] {combo} — already exists, skipping")
                skipped += 1
                continue

            print(f"\n[{done}/{total_combos}] {combo} ({supply_mix})")

            # ── Step 1: Get prices ────────────────────────────────────
            pp = prices_path(zone_label, year)

            # Check if we have bulk prices for 2024
            bulk = os.path.join(ROOT_DIR, BULK_PRICES_2024[zone_label]) if year == 2024 and zone_label in BULK_PRICES_2024 else None
            if bulk and os.path.exists(bulk):
                pp = bulk
                print(f"  Using existing bulk prices: {pp}")
            elif not os.path.exists(pp):
                # Fetch from ENTSO-E
                ok = run([
                    sys.executable, "scripts/fetch_entsoe.py",
                    "--start", f"{year}01010000",
                    "--end", f"{year}12310000",
                    "--zone", zone_eic,
                    "--zone-label", zone_label,
                    "--output-dir", f"./data/entsoe_raw/{zone_label}_{year}",
                    "--prices-only",
                ], f"Fetch ENTSO-E prices: {combo}")

                if not ok:
                    failed.append(f"{combo} (prices)")
                    continue

                time.sleep(2)  # rate limit courtesy

            # ── Step 2: Fetch UMM text + build Parquet ────────────────
            ok = run([
                sys.executable, "scripts/fetch_nordpool_umm.py",
                "--zone", zone_label,
                "--year", str(year),
                "--prices-csv", pp,
            ], f"Fetch Nord Pool UMM + build Parquet: {combo}")

            if not ok:
                failed.append(f"{combo} (umm)")
                continue

            time.sleep(1)

    # ── Step 3: Combine all Parquets ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Combining all Parquet files into training dataset...")
    print(f"{'=' * 60}")

    import pandas as pd

    frames = []
    for path in sorted(glob.glob(os.path.join(ROOT_DIR, "data/entsoe/*.parquet"))):
        if "training" in path:
            continue
        df = pd.read_parquet(path)
        df["series_id"] = os.path.basename(path).replace(".parquet", "")
        frames.append(df)

    if not frames:
        print("  ✗ No Parquet files found!")
        sys.exit(1)

    training = pd.concat(frames).reset_index(drop=True)
    out_path = os.path.join(ROOT_DIR, "data/entsoe/entsoe_training.parquet")
    training.to_parquet(out_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    n_annotated = (training["text"] != "").sum()
    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")
    print(f"  Total rows:          {len(training)}")
    print(f"  Annotated (text!=''):{n_annotated}")
    print(f"  Series:              {training.series_id.nunique()}")
    print(f"  Skipped (existing):  {skipped}")
    print(f"  Failed:              {len(failed)}")
    if failed:
        print(f"    {failed}")
    print(f"\nBreakdown:")
    print(training.groupby("series_id")[["y_t"]].agg(["count", "mean"]).to_string())
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
