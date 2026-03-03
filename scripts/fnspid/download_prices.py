#!/usr/bin/env python3
"""
Download historical stock price data for given symbols using yfinance.

Creates one CSV per symbol with columns: date, close
These CSVs are used by extract_text.py (--history-dir) to align price
data with FNSPID news articles.

Requires: pip install yfinance
"""

import argparse
from pathlib import Path
from datetime import datetime

from tqdm import tqdm


def download_symbol(symbol: str, output_dir: Path, start: str, end: str) -> bool:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required: pip install yfinance")

    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start, end=end, auto_adjust=True)
    if hist.empty:
        print(f"  {symbol}: no data returned")
        return False

    hist = hist.reset_index()
    hist = hist.rename(columns={"Date": "date", "Close": "close"})
    hist["date"] = hist["date"].dt.strftime("%Y-%m-%d")
    hist = hist[["date", "close"]]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{symbol}.csv"
    hist.to_csv(out_path, index=False)
    print(f"  {symbol}: {len(hist)} rows -> {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download stock price history via yfinance"
    )
    parser.add_argument("symbols", nargs="*", help="Stock symbols (e.g., AAPL MSFT)")
    parser.add_argument(
        "--symbols-file", type=str, default=None, help="File with one symbol per line"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="full_history",
        help="Output directory (default: full_history)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2017-01-01",
        help="Start date YYYY-MM-DD (default: 2017-01-01)",
    )
    parser.add_argument(
        "--end", type=str, default=None, help="End date YYYY-MM-DD (default: today)"
    )
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols if s.strip()]
    if args.symbols_file:
        p = Path(args.symbols_file)
        if p.exists():
            symbols.extend(
                s.strip().upper()
                for s in p.read_text().strip().splitlines()
                if s.strip()
            )
    if not symbols:
        parser.error("Provide at least one symbol")

    end = args.end or datetime.now().strftime("%Y-%m-%d")
    output_dir = Path(args.output_dir)

    print(
        f"Downloading {len(symbols)} symbols ({args.start} to {end}) -> {output_dir}/"
    )
    for symbol in tqdm(symbols, desc="Downloading", unit="symbol"):
        download_symbol(symbol, output_dir, args.start, end)

    print("Done.")


if __name__ == "__main__":
    main()
