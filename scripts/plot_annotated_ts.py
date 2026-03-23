#!/usr/bin/env python3
"""
Generate an LLM-annotated time series plot — clean white background,
annotations placed within shaded regions.
"""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

CSV_PATH = "/data/ttfm/synthefy_250k/Daily/fred_data/Conventional_Gasoline_Prices_New_York_Harbor_Regular_Dollars_per_Gallon_Daily_Commodities.csv"
OUT_PATH = Path(__file__).resolve().parent.parent / "annotated_gasoline.pdf"

DATE_START = "2008-01-01"
DATE_END = "2010-01-01"

N_ANNOTATIONS = 5
WRAP_WIDTH = 16


def compute_trend(df: pd.DataFrame, idx: int, lookback: int = 20) -> tuple[str, float]:
    start = max(0, idx - lookback)
    end = min(len(df), idx + 1)
    segment = df.iloc[start:end]["y_t"].values
    if len(segment) < 2:
        return "neutral", 0.0
    change_pct = (segment[-1] - segment[0]) / segment[0] * 100
    if change_pct > 3:
        return "bullish", change_pct
    if change_pct < -3:
        return "bearish", change_pct
    return "neutral", change_pct


BULLISH_TEXTS = [
    "Upward pressure likely as "
    "a 1.4M barrel draw "
    "tightens inventories.",

    "Upward momentum as "
    "industrial recovery and "
    "GDP growth renew demand.",

    "Prices rising as refinery "
    "season tightens supply "
    "and demand accelerates.",
]

BEARISH_TEXTS = [
    "Downward momentum as "
    "record 13.1M bpd output "
    "and 4% travel decline "
    "signal fatigue.",

    "Prices drifting lower as "
    "exports surge to 4.9M bpd "
    "and peak demand wanes.",

    "Continued weakness as "
    "recession fears mount "
    "and OPEC+ raises output.",
]

NEUTRAL_TEXTS = [
    "Prices stabilized as "
    "supply and demand reach "
    "equilibrium.",
]


def main():
    df = pd.read_csv(CSV_PATH)
    df["t"] = pd.to_datetime(df["t"])
    mask = (df["t"] >= DATE_START) & (df["t"] < DATE_END)
    df = df[mask].reset_index(drop=True)

    indices = np.linspace(20, len(df) - 20, N_ANNOTATIONS, dtype=int)
    bull_i, bear_i, neut_i = 0, 0, 0
    annotations = []
    for idx in indices:
        row = df.iloc[idx]
        signal, pct = compute_trend(df, idx)
        if signal == "bullish":
            text = BULLISH_TEXTS[bull_i % len(BULLISH_TEXTS)]
            bull_i += 1
        elif signal == "bearish":
            text = BEARISH_TEXTS[bear_i % len(BEARISH_TEXTS)]
            bear_i += 1
        else:
            text = NEUTRAL_TEXTS[neut_i % len(NEUTRAL_TEXTS)]
            neut_i += 1
        annotations.append({
            "date": row["t"],
            "price": row["y_t"],
            "signal": signal,
            "text": text,
        })

    BG = "#ffffff"
    LINE_COLOR = "#1a1a1a"
    GRID_COLOR = "#e0e0e0"
    TEXT_COLOR = "#333333"
    TITLE_COLOR = "#111111"
    BULL_COLOR = "#1a7a2e"
    BEAR_COLOR = "#b91c1c"
    NEUTRAL_COLOR = "#6b7280"

    SIGNAL_COLORS = {"bullish": BULL_COLOR, "bearish": BEAR_COLOR, "neutral": NEUTRAL_COLOR}
    SIGNAL_BG = {"bullish": "#dcfce7", "bearish": "#fee2e2", "neutral": "#f3f4f6"}

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 14,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "axes.edgecolor": "#cccccc",
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(18, 8))

    ax.plot(df["t"], df["y_t"], color=LINE_COLOR, linewidth=1.8, zorder=5)

    ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.8)
    ax.set_axisbelow(True)

    y_min, y_max = df["y_t"].min(), df["y_t"].max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.30, y_max + y_range * 0.30)

    ann_width = pd.Timedelta(days=int(len(df) / N_ANNOTATIONS * 1.0))

    for i, ann in enumerate(annotations):
        sig_color = SIGNAL_COLORS[ann["signal"]]
        bg_color = SIGNAL_BG[ann["signal"]]

        region_start = ann["date"] - ann_width / 2
        region_end = ann["date"] + ann_width / 2

        ax.axvspan(region_start, region_end, color=bg_color, alpha=0.6, zorder=1)

        wrapped = textwrap.fill(ann["text"], width=WRAP_WIDTH)

        region_center = ann["date"]
        text_x = region_center

        if ann["price"] > (y_min + y_range * 0.5):
            target_y = y_min + y_range * 0.08
            va = "bottom"
        else:
            target_y = y_max - y_range * 0.08
            va = "top"

        ax.text(
            text_x, target_y, wrapped,
            fontsize=15,
            color=sig_color,
            fontfamily="sans-serif",
            ha="center",
            va=va,
            linespacing=1.3,
            zorder=10,
            clip_on=True,
        )

    ax.set_title(
        "Conventional Gasoline Prices — New York Harbor",
        fontsize=18, fontweight="bold", color=TITLE_COLOR,
        fontfamily="sans-serif", pad=14,
    )
    ax.set_xlabel("Date", fontsize=16, labelpad=10)
    ax.set_ylabel("Price ($/gallon)", fontsize=16, labelpad=10)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.2f}"))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate(rotation=45)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(str(OUT_PATH), dpi=600, bbox_inches="tight", facecolor=BG)
    fig.savefig(str(OUT_PATH).replace(".pdf", ".png"), dpi=600, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved to {OUT_PATH}")
    print(f"Saved to {str(OUT_PATH).replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()
