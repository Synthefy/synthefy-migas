#!/usr/bin/env python3
"""
Plot news presence over time from te_news_by_symbol/ to visually choose
periods with most available news for evals.

Reads all JSONs in --input-dir (each has [{"date": "YYYY-MM-DD", "llm_summary": "..."}]),
aggregates counts by day/symbol, and produces:
  - Total news per day (line + rolling mean)
  - Total news per month (bar)
  - Coverage: number of symbols with ≥1 news per day (rolling optional)
  - Heatmap: month x year, color = total news (or symbols-with-news)
  - Per-symbol: one plot per symbol (monthly bars) in --output-dir/by_symbol/
  - Symbol grid: one PNG with subplots for top N symbols (monthly counts)

Usage:
  uv run python scripts/plot_te_news_presence.py --input-dir data/te_commodities/te_news_by_symbol --output-dir data/te_commodities/te_news_plots
  uv run python scripts/plot_te_news_presence.py --per-symbol --grid --grid-size 48
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_news_by_symbol(input_dir: Path) -> pd.DataFrame:
    """Load all te_news_by_symbol/*.json into a DataFrame with columns: date, symbol, count (1 per row)."""
    rows = []
    for path in sorted(input_dir.glob("*.json")):
        stem = path.stem
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for item in data:
            d = item.get("date") or item.get("Date")
            if not d:
                continue
            rows.append({"date": d, "symbol": stem})
    if not rows:
        return pd.DataFrame(columns=["date", "symbol"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def plot_total_per_day(df: pd.DataFrame, output_dir: Path, rolling_days: int = 30) -> None:
    """Total news per day + rolling mean."""
    daily = df.groupby("date").size().reset_index(name="count")
    daily = daily.sort_values("date")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(daily["date"], daily["count"], alpha=0.3, color="steelblue")
    ax.plot(daily["date"], daily["count"], color="steelblue", linewidth=0.8, label="daily count")
    if rolling_days and len(daily) >= rolling_days:
        daily["rolling"] = daily["count"].rolling(rolling_days, center=True).mean()
        ax.plot(daily["date"], daily["rolling"], color="darkblue", linewidth=2, label=f"{rolling_days}-day rolling")
    ax.set_xlabel("Date")
    ax.set_ylabel("News count")
    ax.set_title("Total commodity news per day")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "news_per_day.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_dir / 'news_per_day.png'}")


def plot_total_per_month(df: pd.DataFrame, output_dir: Path) -> None:
    """Total news per month (bar)."""
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month").size()
    monthly.index = monthly.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(monthly.index, monthly.values, width=20, color="steelblue", alpha=0.8, edgecolor="none")
    ax.set_xlabel("Month")
    ax.set_ylabel("News count")
    ax.set_title("Total commodity news per month")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "news_per_month.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_dir / 'news_per_month.png'}")


def plot_coverage_per_day(df: pd.DataFrame, output_dir: Path, rolling_days: int = 30) -> None:
    """Number of symbols with ≥1 news per day (coverage)."""
    daily_symbols = df.groupby("date")["symbol"].nunique().reset_index(name="symbols")
    daily_symbols = daily_symbols.sort_values("date")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(daily_symbols["date"], daily_symbols["symbols"], alpha=0.3, color="forestgreen")
    ax.plot(daily_symbols["date"], daily_symbols["symbols"], color="forestgreen", linewidth=0.8, label="daily coverage")
    if rolling_days and len(daily_symbols) >= rolling_days:
        daily_symbols = daily_symbols.copy()
        daily_symbols["rolling"] = daily_symbols["symbols"].rolling(rolling_days, center=True).mean()
        ax.plot(daily_symbols["date"], daily_symbols["rolling"], color="darkgreen", linewidth=2, label=f"{rolling_days}-day rolling")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of symbols with ≥1 news")
    ax.set_title("Commodity news coverage (symbols with news per day)")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_per_day.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_dir / 'coverage_per_day.png'}")


def plot_heatmap_monthly(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap: year x month, color = total news in that month."""
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    grid = df.groupby(["year", "month"]).size().unstack(fill_value=0)
    for m in range(1, 13):
        if m not in grid.columns:
            grid[m] = 0
    grid = grid.reindex(columns=sorted(grid.columns)).sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(4, len(grid) * 0.35)))
    im = ax.imshow(grid.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(grid)))
    ax.set_yticklabels(grid.index.astype(int))
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_title("Total commodity news per month (heatmap)")
    plt.colorbar(im, ax=ax, label="News count")
    fig.tight_layout()
    fig.savefig(output_dir / "heatmap_news_by_month.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_dir / 'heatmap_news_by_month.png'}")


def plot_coverage_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap: year x month, color = number of symbols with ≥1 news in that month."""
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    coverage = df.groupby(["year", "month"])["symbol"].nunique().unstack(fill_value=0)
    for m in range(1, 13):
        if m not in coverage.columns:
            coverage[m] = 0
    coverage = coverage.reindex(columns=sorted(coverage.columns)).sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(4, len(coverage) * 0.35)))
    im = ax.imshow(coverage.values, aspect="auto", cmap="Greens", interpolation="nearest")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(coverage)))
    ax.set_yticklabels(coverage.index.astype(int))
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_title("Symbols with ≥1 news per month (coverage heatmap)")
    plt.colorbar(im, ax=ax, label="Number of symbols")
    fig.tight_layout()
    fig.savefig(output_dir / "heatmap_coverage_by_month.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_dir / 'heatmap_coverage_by_month.png'}")


def plot_per_symbol_monthly(
    df: pd.DataFrame, output_dir: Path, input_dir: Path, subdir: str = "by_symbol"
) -> None:
    """One PNG per symbol: monthly news count (bar chart). Includes symbols with 0 news (from input_dir)."""
    out = output_dir / subdir
    out.mkdir(parents=True, exist_ok=True)
    all_stems = sorted({p.stem for p in input_dir.glob("*.json")})
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    for sym in all_stems:
        sub = df[df["symbol"] == sym].groupby("month").size().reset_index(name="count")
        sub = sub.sort_values("month")
        fig, ax = plt.subplots(figsize=(10, 3))
        if len(sub) > 0:
            ax.bar(sub["month"], sub["count"], width=15, color="steelblue", alpha=0.85, edgecolor="none")
        ax.set_xlabel("Month")
        ax.set_ylabel("News count")
        ax.set_title(f"{sym}")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)
        fig.tight_layout()
        fig.savefig(out / f"{sym}.png", dpi=150, bbox_inches="tight")
        plt.close()
    print(f"  Saved {len(all_stems)} per-symbol plots to {out}/")


def plot_symbol_grid(df: pd.DataFrame, output_dir: Path, top_n: int = 48) -> None:
    """One PNG with a grid of subplots: each subplot = one symbol's monthly news count (top N by total news)."""
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    total_per_sym = df.groupby("symbol").size().sort_values(ascending=False)
    symbols = total_per_sym.head(top_n).index.tolist()
    n = len(symbols)
    n_cols = 8
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 1.8 * n_rows), squeeze=False)
    for idx, sym in enumerate(symbols):
        ax = axes.flat[idx]
        sub = df[df["symbol"] == sym].groupby("month").size().reset_index(name="count")
        sub = sub.sort_values("month")
        ax.bar(sub["month"], sub["count"], width=12, color="steelblue", alpha=0.85, edgecolor="none")
        ax.set_title(sym, fontsize=9)
        ax.tick_params(axis="both", labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    for j in range(len(symbols), n_rows * n_cols):
        axes.flat[j].set_visible(False)
    fig.suptitle(f"News per month (top {n} symbols by total news)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "grid_per_symbol_monthly.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_dir / 'grid_per_symbol_monthly.png'} (top {n} symbols)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot TE commodity news presence over time for choosing eval periods"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/te_commodities/te_news_by_symbol",
        help="Directory of per-symbol JSONs (date + llm_summary)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/te_commodities/te_news_plots",
        help="Where to save PNGs",
    )
    parser.add_argument(
        "--rolling-days",
        type=int,
        default=30,
        help="Rolling window in days for smoothing (default 30)",
    )
    parser.add_argument(
        "--per-symbol",
        action="store_true",
        help="Write one PNG per symbol (monthly bars) to output-dir/by_symbol/",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Write one grid PNG with subplots for top N symbols (--grid-size).",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=48,
        help="Number of symbols in the grid plot (default 48).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading news from {input_dir} ...")
    df = load_news_by_symbol(input_dir)
    if df.empty:
        raise SystemExit("No news records found.")
    print(f"  Loaded {len(df)} records, {df['symbol'].nunique()} symbols, date range {df['date'].min().date()} to {df['date'].max().date()}")

    print("Generating plots...")
    plot_total_per_day(df, output_dir, rolling_days=args.rolling_days)
    plot_total_per_month(df, output_dir)
    plot_coverage_per_day(df, output_dir, rolling_days=args.rolling_days)
    plot_heatmap_monthly(df, output_dir)
    plot_coverage_heatmap(df, output_dir)
    if args.per_symbol:
        plot_per_symbol_monthly(df, output_dir, input_dir)
    if args.grid:
        plot_symbol_grid(df, output_dir, top_n=args.grid_size)

    print(f"\nPlots saved to {output_dir}. Use heatmaps and coverage plots to pick periods with most news for evals.")


if __name__ == "__main__":
    main()
