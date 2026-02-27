#!/usr/bin/env python3
"""
Plot daily text presence for each (country, symbol) from daily_presence CSVs.

Reads data/te_countries/daily_presence/*.csv (t, text_present) and draws a timeline:
- Optional thin bar or step: 1 when that day has news, 0 otherwise.
- Optional rolling window (e.g. 30-day) fraction of days with news to spot consistent periods.

Saves one PNG per series to data/te_countries/daily_presence_plots/{stem}.png so you can
visually choose date ranges with consistent text. Use --yearly to get one plot per year per stem
(saved in output_dir/yearly/{stem}_{year}.png) for a less crowded view.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def sanitize_stem(country: str, symbol: str) -> str:
    c = re.sub(r"[^\w\s]", "", country).strip().replace(" ", "_")
    s = re.sub(r"[^\w]", "_", symbol.strip())
    return f"{c}_{s}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot daily text presence timelines for (country, symbol) series",
    )
    parser.add_argument(
        "--presence-dir",
        type=Path,
        default=Path("data/te_countries/daily_presence"),
        help="Directory of daily presence CSVs (t, text_present)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/te_countries/daily_presence_plots"),
        help="Directory for output PNGs",
    )
    parser.add_argument(
        "--ranked-csv",
        type=Path,
        default=Path("data/te_countries/exploration_ranked.csv"),
        help="Optional: ranked CSV to get (country, symbol) order and titles; if not set, plot all CSVs in presence-dir",
    )
    parser.add_argument(
        "--rolling",
        type=int,
        default=30,
        help="Rolling window (days) for smoothed presence fraction; 0 to disable (default 30)",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Plot only this stem (e.g. United_States_SPX); default: all",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Output PNG DPI (default 120)",
    )
    parser.add_argument(
        "--yearly",
        action="store_true",
        help="Plot one PNG per year per stem (saved in output-dir/yearly/) so each year is less crowded.",
    )
    args = parser.parse_args()

    if not args.presence_dir.is_dir():
        raise SystemExit(f"Presence dir not found: {args.presence_dir}")

    stems = []
    if args.ranked_csv.exists():
        df = pd.read_csv(args.ranked_csv)
        for _, row in df.iterrows():
            stem = sanitize_stem(str(row["country"]), str(row["symbol"]))
            stems.append(stem)
    else:
        stems = [p.stem for p in args.presence_dir.glob("*.csv")]

    if args.stem:
        stems = [s for s in stems if s == args.stem]
    if not stems:
        print("No stems to plot.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.yearly:
        (args.output_dir / "yearly").mkdir(parents=True, exist_ok=True)

    for stem in stems:
        csv_path = args.presence_dir / f"{stem}.csv"
        if not csv_path.exists():
            print(f"Skip {stem}: no CSV")
            continue
        df = pd.read_csv(csv_path)
        df["t"] = pd.to_datetime(df["t"])
        df = df.sort_values("t")

        if args.yearly:
            years = sorted(df["t"].dt.year.dropna().unique().astype(int))
            for year in years:
                sub = df[df["t"].dt.year == year].copy()
                if len(sub) == 0:
                    continue
                sub = sub.sort_values("t")
                y = sub["text_present"].astype(int)
                fig, ax = plt.subplots(figsize=(12, 2.5))
                ax.fill_between(sub["t"], 0, y, step="post", alpha=0.7, color="steelblue")
                if args.rolling and len(sub) >= args.rolling:
                    roll = sub["text_present"].rolling(args.rolling, center=True).mean()
                    ax.plot(sub["t"], roll, color="darkblue", linewidth=1.2, label=f"{args.rolling}-day share")
                    ax.legend(loc="upper right", fontsize=8)
                ax.set_ylim(-0.05, 1.15)
                ax.set_ylabel("Text present")
                ax.set_xlabel("Date")
                ax.set_title(f"{stem.replace('_', ' ')} — {year}")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                out_path = args.output_dir / "yearly" / f"{stem}_{year}.png"
                fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
                plt.close(fig)
            print(f"  {stem} -> {len(years)} yearly PNGs in {args.output_dir / 'yearly'}")
        else:
            y = df["text_present"].astype(int)
            fig, ax = plt.subplots(figsize=(12, 2.5))
            ax.fill_between(df["t"], 0, y, step="post", alpha=0.7, color="steelblue")
            if args.rolling and len(df) >= args.rolling:
                roll = df["text_present"].rolling(args.rolling, center=True).mean()
                ax.plot(df["t"], roll, color="darkblue", linewidth=1.2, label=f"{args.rolling}-day share with news")
                ax.legend(loc="upper right", fontsize=8)
            ax.set_ylim(-0.05, 1.15)
            ax.set_ylabel("Text present")
            ax.set_xlabel("Date")
            ax.set_title(f"{stem.replace('_', ' ')} — daily news presence")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out_path = args.output_dir / f"{stem}.png"
            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"  {stem} -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
