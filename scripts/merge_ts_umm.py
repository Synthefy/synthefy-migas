"""
Merge ENTSO-E time series with Nord Pool UMM text annotations.

For each timestamp in the time series, finds all UMM events where
event_start <= timestamp <= event_end and creates a text_annotation column.
Multiple overlapping events are labeled "Event 1: ..., Event 2: ...", etc.

Usage:
  uv run python scripts/merge_ts_umm.py --zone SE_1 --year 2022
  uv run python scripts/merge_ts_umm.py --all
  uv run python scripts/merge_ts_umm.py --all --input-dir data/all --output-dir data/merged
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def merge_ts_umm(ts_path: Path, umm_path: Path) -> pd.DataFrame:
    """Merge a time series file with its corresponding UMM file."""
    ts = pd.read_parquet(ts_path)
    umm = pd.read_parquet(umm_path)

    # Normalize both to UTC for comparison
    ts_col = "timestamp" if "timestamp" in ts.columns else "date"
    ts["_ts_utc"] = pd.to_datetime(ts[ts_col], utc=True)
    umm["_start_utc"] = pd.to_datetime(umm["event_start"], utc=True)
    umm["_end_utc"] = pd.to_datetime(umm["event_end"], utc=True)

    EVENT_FIELDS = [
        "unavailability_type", "reason_text", "remarks", "asset_name",
        "fuel_type", "installed_capacity_mw", "unavailable_mw", "text",
    ]

    # For each timestamp, find overlapping events
    ts_utc = ts["_ts_utc"]
    annotations = [[] for _ in range(len(ts))]

    for _, event in umm.iterrows():
        start = event["_start_utc"]
        end = event["_end_utc"]
        if pd.isna(start) or pd.isna(end):
            continue

        entry = {}
        for field in EVENT_FIELDS:
            val = event.get(field)
            if pd.notna(val) and val != "":
                entry[field] = val if not isinstance(val, float) else round(val, 1)
        if not entry:
            continue

        mask = (ts_utc >= start) & (ts_utc <= end)
        for i in ts.index[mask]:
            annotations[i].append(entry)

    # Serialize as JSON string — empty list becomes ""
    ts["text_annotation"] = [
        json.dumps(events, ensure_ascii=False) if events else ""
        for events in annotations
    ]
    ts = ts.drop(columns=["_ts_utc"])
    return ts


def main():
    parser = argparse.ArgumentParser(
        description="Merge ENTSO-E time series with Nord Pool UMM text annotations."
    )
    parser.add_argument("--zone", help="Zone label (e.g. SE_1)")
    parser.add_argument("--year", type=int, help="Year (e.g. 2022)")
    parser.add_argument("--all", action="store_true",
                        help="Merge all matching ts/umm pairs found in input dir")
    parser.add_argument("--input-dir", default="data/all",
                        help="Directory with _ts and _umm files (default: data/all)")
    parser.add_argument("--output-dir", default="data/merged",
                        help="Output directory (default: data/merged)")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        # Find all _ts files that have a matching _umm file
        ts_files = sorted(in_dir.glob("*_ts.parquet"))
        pairs = []
        for ts_path in ts_files:
            umm_path = in_dir / ts_path.name.replace("_ts.parquet", "_umm.parquet")
            if umm_path.exists():
                pairs.append((ts_path, umm_path))

        pbar = tqdm(pairs, desc="Merging", unit="file")
        for ts_path, umm_path in pbar:
            name = ts_path.stem.replace("_ts", "")
            pbar.set_postfix_str(name)
            merged = merge_ts_umm(ts_path, umm_path)
            merged.to_parquet(out_dir / f"{name}.parquet", index=False)
            merged.to_csv(out_dir / f"{name}.csv", index=False)
            n_annotated = (merged["text_annotation"] != "").sum()
            tqdm.write(f"  {name}: {len(merged)} rows, {n_annotated} annotated")
    elif args.zone and args.year:
        ts_path = in_dir / f"{args.zone}_{args.year}_ts.parquet"
        umm_path = in_dir / f"{args.zone}_{args.year}_umm.parquet"
        if not ts_path.exists():
            print(f"Not found: {ts_path}")
            return
        if not umm_path.exists():
            print(f"Not found: {umm_path}")
            return

        name = f"{args.zone}_{args.year}"
        print(f"Merging {name}...")
        merged = merge_ts_umm(ts_path, umm_path)
        merged.to_parquet(out_dir / f"{name}.parquet", index=False)
        merged.to_csv(out_dir / f"{name}.csv", index=False)
        n_annotated = (merged["text_annotation"] != "").sum()
        print(f"  {len(merged)} rows, {n_annotated} annotated → {out_dir / name}.parquet")
    else:
        parser.error("Provide --zone and --year, or use --all")


if __name__ == "__main__":
    main()
