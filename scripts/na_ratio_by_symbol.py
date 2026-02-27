import json
import argparse
from pathlib import Path

import pandas as pd

import ipdb

def main():
    p = argparse.ArgumentParser(description="Compute NA ratio in text column per symbol CSV.")
    p.add_argument("--dir", type=str, default="/home/ubuntu/bekzat/FNSPID_3k", help="Directory with *_with_text.csv files")
    p.add_argument("--output", "-o", type=str, default="na_ratio_by_symbol.json", help="Output JSON path")
    args = p.parse_args()

    dir_path = Path(args.dir)
    symbol_to_na_ratio = {}

    for csv_path in sorted(dir_path.glob("*_with_text.csv")):
        symbol = csv_path.stem.replace("_with_text", "")
        df = pd.read_csv(csv_path)
        if "text" not in df.columns:
            symbol_to_na_ratio[symbol] = None  # JSON-serializable
            continue
        n_total = len(df)
        text = df["text"]
        s = text.astype(str).str.strip()
        is_na = text.isna() | (s == "NA") | s.str.lower().str.contains("error", na=False)
        n_na = is_na.sum()
        symbol_to_na_ratio[symbol] = round(n_na / n_total, 6) if n_total else 0.0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(symbol_to_na_ratio, f, indent=2)

    print(f"Saved {len(symbol_to_na_ratio)} symbols to {out_path}")
    ipdb.set_trace()

 
if __name__ == "__main__":
    main()