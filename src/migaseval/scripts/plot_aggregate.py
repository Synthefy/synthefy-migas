#!/usr/bin/env python3
"""
Generate an aggregate multi-page PDF summarising evaluation results across
context lengths.

Reads prediction .npz files produced by ``migaseval.evaluation`` and computes
metrics directly.  Optionally filters out windows with bad LLM summaries when
a summaries directory is provided.

Contents:
  0. Summary quality statistics (if summaries_dir given)
  1. Aggregate MAE / MSE per context length
  2. Win counts for Migas-1.5 vs each baseline
  3. Elo rating plots (overall bar + by-context-length line)
  4. Per context-length tables with individual dataset results

Usage (standalone):
    python -m migaseval.scripts.plot_aggregate --output_dir ./results
    python -m migaseval.scripts.plot_aggregate --output_dir ./results --summaries_dir ./data/test/summaries
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from migaseval.eval_utils import (
    MODEL_COLORS,
    MODEL_DISPLAY_NAMES,
    MODEL_ORDER,
    OURS_MODELS,
    compute_metrics,
    get_display_name,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.size"] = 10

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Summary quality ──────────────────────────────────────────────────────────


def _is_useful_summary(text: str) -> bool:
    """A summary is useful only if it contains both a facts section and a
    predictive-signals section."""
    upper = text.upper()
    has_facts = "FACTS:" in upper or "FACTUAL SUMMARY" in upper
    has_signals = "PREDICTIVE SIGNALS" in upper or "PREDICTIVE SIGNAL:" in upper
    return has_facts and has_signals


def _load_useful_mask(
    summaries_dir: str, ds_name: str, n_samples: int
) -> np.ndarray:
    """Return a boolean mask of length *n_samples*.  ``True`` means the window
    has a useful summary and should be included in metric computation."""
    ds_dir = os.path.join(summaries_dir, ds_name)
    if not os.path.isdir(ds_dir):
        return np.ones(n_samples, dtype=bool)

    mask = np.zeros(n_samples, dtype=bool)
    for idx in range(n_samples):
        path = os.path.join(ds_dir, f"summary_{idx}.json")
        if not os.path.isfile(path):
            mask[idx] = True  # no summary file → include
            continue
        with open(path) as fh:
            data = json.load(fh)
        if _is_useful_summary(data.get("summary", "")):
            mask[idx] = True
    return mask


# ── Data loading ─────────────────────────────────────────────────────────────


def _discover_context_lengths(output_dir: Path) -> list[int]:
    """Find context_<N> subdirectories that contain a predictions/ folder."""
    ctx_lengths = []
    for d in sorted(output_dir.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"context_(\d+)$", d.name)
        if m and (d / "predictions").is_dir():
            ctx_lengths.append(int(m.group(1)))
    return sorted(ctx_lengths)


def _discover_datasets(pred_dir: Path) -> list[str]:
    """List dataset subdirectories under a predictions/ folder."""
    if not pred_dir.is_dir():
        return []
    return sorted(d.name for d in pred_dir.iterdir() if d.is_dir())


def _discover_models(pred_dir: Path, ds_name: str) -> list[str]:
    """List model keys from .npz files in a dataset predictions directory."""
    ds_dir = pred_dir / ds_name
    if not ds_dir.is_dir():
        return []
    return sorted(p.stem for p in ds_dir.glob("*.npz"))


def load_all_data(
    output_dir: Path,
    summaries_dir: str | None = None,
    models: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Walk every (context_length, dataset) and compute metrics from stored
    prediction arrays.

    Returns (metrics_df, summary_stats_df).  summary_stats_df is None when
    summaries_dir is not provided.
    """
    if models is None:
        models = MODEL_ORDER

    ctx_lengths = _discover_context_lengths(output_dir)
    if not ctx_lengths:
        raise RuntimeError(f"No context_*/predictions/ directories found in {output_dir}")

    records: list[dict] = []
    summary_stats: list[dict] = []

    for ctx in ctx_lengths:
        pred_dir = output_dir / f"context_{ctx}" / "predictions"
        ds_names = _discover_datasets(pred_dir)

        for ds_name in ds_names:
            # Find a reference npz to get sample count
            available = _discover_models(pred_dir, ds_name)
            ref_model = None
            for m in models:
                if m in available:
                    ref_model = m
                    break
            if ref_model is None:
                continue

            ref_npz = pred_dir / ds_name / f"{ref_model}.npz"
            ref_data = np.load(ref_npz)
            n_total = int(ref_data["gt"].shape[0])

            # Summary quality mask
            if summaries_dir:
                mask = _load_useful_mask(summaries_dir, ds_name, n_total)
                n_useful = int(mask.sum())
                summary_stats.append(
                    {
                        "context_length": ctx,
                        "dataset": ds_name,
                        "n_total": n_total,
                        "n_useful": n_useful,
                        "n_bad": n_total - n_useful,
                    }
                )
            else:
                mask = np.ones(n_total, dtype=bool)
                n_useful = n_total

            if n_useful == 0:
                continue

            row: dict = {
                "context_length": ctx,
                "dataset": ds_name,
                "n_samples": n_useful,
            }

            for m in models:
                npz_path = pred_dir / ds_name / f"{m}.npz"
                if not npz_path.is_file():
                    continue
                data = np.load(npz_path)
                preds = data["predictions"][mask]
                gt = data["gt"][mask]
                if len(gt) == 0:
                    continue
                metrics = compute_metrics(preds, gt)
                row[f"{m}_mean_mae"] = metrics["mean_mae"]
                row[f"{m}_mean_mse"] = metrics["mean_mse"]

            records.append(row)

    if not records:
        raise RuntimeError(f"No prediction npz files found in {output_dir}")

    metrics_df = pd.DataFrame(records)
    stats_df = pd.DataFrame(summary_stats) if summary_stats else None
    return metrics_df, stats_df


def available_models(df: pd.DataFrame, models: list[str] | None = None) -> list[str]:
    if models is None:
        models = MODEL_ORDER
    return [
        m
        for m in models
        if f"{m}_mean_mae" in df.columns and df[f"{m}_mean_mae"].notna().any()
    ]


# ── Elo ──────────────────────────────────────────────────────────────────────


def _get_rankings(df: pd.DataFrame, models: list[str]) -> list[list[tuple[str, float]]]:
    rankings = []
    for _, row in df.iterrows():
        vals = []
        for m in models:
            col = f"{m}_mean_mae"
            if col in row.index and pd.notna(row[col]):
                vals.append((m, row[col]))
        if len(vals) >= 2:
            vals.sort(key=lambda x: x[1])
            rankings.append(vals)
    return rankings


def compute_elo(
    rankings: list, models: list[str], base: int = 1500, n_seeds: int = 32
) -> dict[str, int]:
    """Multi-seed Elo computation.  Uses multielo if available, otherwise a
    simple pairwise update."""
    if not rankings:
        return {m: base for m in models}

    try:
        from multielo import MultiElo

        elo_calc = MultiElo(k_value=32, d_value=400)
        use_multielo = True
    except ImportError:
        use_multielo = False

    all_ratings: dict[str, list[float]] = {m: [] for m in models}

    for seed in range(n_seeds):
        ratings = {m: float(base) for m in models}
        shuffled = list(rankings)
        random.seed(42 + seed)
        random.shuffle(shuffled)

        for ranking in shuffled:
            if len(ranking) < 2:
                continue
            order = [m for m, _ in ranking]

            if use_multielo:
                current = np.array([ratings[m] for m in order])
                new = elo_calc.get_new_ratings(current)
                for m, r in zip(order, new):
                    ratings[m] = float(r)
            else:
                # Simple pairwise K=32 update
                k = 32.0
                for i in range(len(order)):
                    for j in range(i + 1, len(order)):
                        wi, wj = order[i], order[j]
                        ri, rj = ratings[wi], ratings[wj]
                        ei = 1.0 / (1.0 + 10 ** ((rj - ri) / 400))
                        ratings[wi] += k * (1.0 - ei)
                        ratings[wj] += k * (0.0 - (1.0 - ei))

        for m in models:
            all_ratings[m].append(ratings[m])

    return {m: int(round(np.mean(all_ratings[m]))) for m in models}


# ── Table rendering helpers ──────────────────────────────────────────────────

TABLE_HEADER_COLOR = "#3b5998"
ALT_ROW_COLORS = ("#f8f9fa", "#ffffff")
BEST_COLOR = "#E87511"
SECOND_COLOR = "#F4B87A"


def _render_table(
    ax,
    col_labels,
    row_data,
    title,
    row_colors=None,
    col_widths=None,
    fontsize=8,
    highlight_groups=None,
):
    """Draw a table on *ax*."""
    ax.axis("off")
    ax.set_title(title, fontsize=fontsize + 4, fontweight="bold", pad=12)

    n_cols = len(col_labels)
    n_rows = len(row_data)
    if n_rows == 0:
        return

    cell_text = []
    cell_colors = []
    for ri, row in enumerate(row_data):
        text_row = []
        color_row = [None] * len(row)
        bg = row_colors[ri] if row_colors else ALT_ROW_COLORS[ri % 2]

        groups = (
            [list(range(1, len(row)))]
            if highlight_groups is None
            else highlight_groups
        )

        for grp in groups:
            numeric_vals = {}
            for ci in grp:
                if ci >= len(row):
                    continue
                try:
                    numeric_vals[ci] = float(row[ci])
                except (ValueError, TypeError):
                    pass
            best_ci = (
                min(numeric_vals, key=numeric_vals.get) if numeric_vals else None
            )
            remaining = {k: v for k, v in numeric_vals.items() if k != best_ci}
            second_ci = (
                min(remaining, key=remaining.get) if remaining else None
            )
            for ci in grp:
                if ci >= len(row):
                    continue
                if ci == best_ci:
                    color_row[ci] = BEST_COLOR
                elif ci == second_ci:
                    color_row[ci] = SECOND_COLOR

        for ci, val in enumerate(row):
            if color_row[ci] is None:
                color_row[ci] = bg
            if isinstance(val, float):
                text_row.append(f"{val:.4f}")
            else:
                text_row.append(str(val))

        cell_text.append(text_row)
        cell_colors.append(color_row)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=[TABLE_HEADER_COLOR] * n_cols,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    if col_widths:
        for ci, w in enumerate(col_widths):
            for ri in range(-1, n_rows):
                table[ri + 1, ci].set_width(w)
    table.scale(1, 1.4)

    for ci in range(n_cols):
        table[0, ci].set_text_props(color="white", fontweight="bold")

    for ri in range(n_rows):
        table[ri + 1, 0].set_text_props(fontweight="bold", ha="left")


# ── PDF pages ────────────────────────────────────────────────────────────────


def page_summary_quality(pdf: PdfPages, stats_df: pd.DataFrame):
    """Page: summary quality statistics."""
    # Deduplicate across context lengths (summaries are shared)
    first_ctx = stats_df["context_length"].min()
    dedup = stats_df[stats_df["context_length"] == first_ctx]

    total = int(dedup["n_total"].sum())
    useful = int(dedup["n_useful"].sum())
    bad = total - useful
    pct = useful / total * 100 if total else 0

    agg_rows = [["All datasets", str(total), str(useful), str(bad), f"{pct:.1f}%"]]
    col_labels = ["Scope", "Total Windows", "Useful", "Bad", "Useful %"]
    row_colors = [ALT_ROW_COLORS[0]]

    # Per-dataset breakdown (only datasets with bad > 0)
    bad_ds = dedup[dedup["n_bad"] > 0].sort_values("n_bad", ascending=False)
    detail_rows = []
    for _, r in bad_ds.iterrows():
        ds_pct = r["n_useful"] / r["n_total"] * 100 if r["n_total"] else 0
        ds = str(r["dataset"]).replace("_with_text", "")
        detail_rows.append(
            [
                ds,
                str(int(r["n_total"])),
                str(int(r["n_useful"])),
                str(int(r["n_bad"])),
                f"{ds_pct:.1f}%",
            ]
        )

    n_subplots = 2 if detail_rows else 1
    heights = (
        [len(agg_rows) + 2, max(len(detail_rows), 1) + 2]
        if n_subplots == 2
        else [len(agg_rows) + 2]
    )
    fig, axes = plt.subplots(
        n_subplots,
        1,
        figsize=(16, sum(heights) * 0.55 + 4),
        gridspec_kw={"height_ratios": heights, "hspace": 0.5},
    )
    if n_subplots == 1:
        axes = [axes]

    _render_table(
        axes[0],
        col_labels,
        agg_rows,
        "Summary Quality — Aggregate",
        row_colors=row_colors,
        fontsize=10,
        highlight_groups=[],
    )

    if detail_rows:
        det_labels = ["Dataset", "Total", "Useful", "Bad", "Useful %"]
        det_colors = [ALT_ROW_COLORS[i % 2] for i in range(len(detail_rows))]
        _render_table(
            axes[1],
            det_labels,
            detail_rows,
            "Datasets with Bad Summaries (excluded from metrics)",
            row_colors=det_colors,
            fontsize=9,
            highlight_groups=[],
        )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_aggregate_metrics(
    pdf: PdfPages,
    df: pd.DataFrame,
    models: list[str],
    ctx_lengths: list[int],
):
    """Separate MAE and MSE aggregate tables per context length."""
    model_labels = [get_display_name(m) for m in models]
    n_model = len(models)

    for metric_suffix, metric_name in [("mean_mae", "MAE"), ("mean_mse", "MSE")]:
        col_labels = ["Ctx"] + model_labels
        rows = []
        row_colors = []
        for ci, ctx in enumerate(ctx_lengths):
            sub = df[df["context_length"] == ctx]
            if sub.empty:
                continue
            row: list = [str(ctx)]
            for m in models:
                col = f"{m}_{metric_suffix}"
                row.append(sub[col].mean() if col in sub.columns else "")
            rows.append(row)
            row_colors.append(ALT_ROW_COLORS[ci % 2])

        ctx_w = 0.06
        metric_w = (1.0 - ctx_w) / n_model
        col_widths = [ctx_w] + [metric_w] * n_model
        highlight = [list(range(1, 1 + n_model))]

        fig, ax = plt.subplots(figsize=(18, max(6, len(rows) * 0.4 + 3)))
        _render_table(
            ax,
            col_labels,
            rows,
            f"Aggregate {metric_name} per Context Length",
            row_colors=row_colors,
            col_widths=col_widths,
            fontsize=9,
            highlight_groups=highlight,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def page_win_counts(
    pdf: PdfPages,
    df: pd.DataFrame,
    models: list[str],
    ctx_lengths: list[int],
):
    """Win counts — how often Migas-1.5 beats each baseline."""
    baselines = [m for m in models if m not in OURS_MODELS]
    our = [m for m in models if m in OURS_MODELS]

    bl_labels = [get_display_name(b) for b in baselines]
    col_labels = ["Our Model", "Ctx", "N"] + [f"vs {l}" for l in bl_labels]

    rows = []
    row_colors = []
    for oi, om in enumerate(our):
        our_col = f"{om}_mean_mae"
        for ctx in ctx_lengths:
            sub = df[df["context_length"] == ctx]
            if sub.empty or our_col not in sub.columns:
                continue
            n_datasets = len(sub)
            row: list = [get_display_name(om), str(ctx), str(n_datasets)]
            for bl in baselines:
                bl_col = f"{bl}_mean_mae"
                if bl_col not in sub.columns:
                    row.append("-")
                    continue
                valid = sub[[our_col, bl_col]].dropna()
                wins = int((valid[our_col] < valid[bl_col]).sum())
                total = len(valid)
                pct = wins / total * 100 if total else 0
                row.append(f"{wins}/{total} ({pct:.0f}%)")
            rows.append(row)
            row_colors.append(ALT_ROW_COLORS[oi % 2])

    n_bl = len(baselines)
    first_w = 0.10
    ctx_w = 0.04
    n_w = 0.04
    bl_w = (1.0 - first_w - ctx_w - n_w) / max(n_bl, 1)
    col_widths = [first_w, ctx_w, n_w] + [bl_w] * n_bl

    fig, ax = plt.subplots(figsize=(20, max(6, len(rows) * 0.40 + 3)))
    _render_table(
        ax,
        col_labels,
        rows,
        "Win Counts: Migas-1.5 vs Baselines per Context Length",
        row_colors=row_colors,
        col_widths=col_widths,
        fontsize=8,
        highlight_groups=[],
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_elo(
    pdf: PdfPages,
    df: pd.DataFrame,
    models: list[str],
    ctx_lengths: list[int],
):
    """Overall Elo bar chart + Elo-by-context-length line plot."""
    fig, axes = plt.subplots(
        2, 1, figsize=(16, 14), gridspec_kw={"hspace": 0.40}
    )

    rankings = _get_rankings(df, models)
    elo = compute_elo(rankings, models)

    sorted_m = sorted(models, key=lambda m: elo[m], reverse=True)
    labels = [get_display_name(m) for m in sorted_m]
    ratings = [elo[m] for m in sorted_m]
    colors = [MODEL_COLORS.get(m, "#999999") for m in sorted_m]

    ax = axes[0]
    x = np.arange(len(sorted_m)) * 1.4
    bars = ax.bar(
        x, ratings, color=colors, edgecolor="#333", linewidth=0.8, width=0.6
    )
    for bar, r in zip(bars, ratings):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(r),
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )
    ax.set_ylabel("Elo Rating", fontweight="bold", fontsize=16)
    ax.set_title("Overall Elo Ratings (Mean MAE)", fontweight="bold", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight="bold")
    ax.set_ylim(min(1350, min(ratings) - 50), max(ratings) + 80)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_axisbelow(True)

    ax2 = axes[1]
    elo_by_ctx: dict[str, list[int]] = {m: [] for m in models}
    for ctx in ctx_lengths:
        sub = df[df["context_length"] == ctx]
        r = _get_rankings(sub, models)
        e = compute_elo(r, models)
        for m in models:
            elo_by_ctx[m].append(e[m])

    for m in models:
        lw = 3.0 if m in OURS_MODELS else 2.0
        marker = "o" if m in OURS_MODELS else "s"
        ax2.plot(
            ctx_lengths,
            elo_by_ctx[m],
            marker=marker,
            linewidth=lw,
            label=get_display_name(m),
            color=MODEL_COLORS.get(m, "#999999"),
            markersize=8,
        )
    ax2.axhline(
        y=1500, color="red", linestyle="--", linewidth=1.0, alpha=0.4, label="Base (1500)"
    )
    ax2.set_xlabel("Context Length", fontweight="bold", fontsize=16)
    ax2.set_ylabel("Elo Rating", fontweight="bold", fontsize=16)
    ax2.set_title("Elo Rating by Context Length", fontweight="bold", fontsize=18)
    ax2.set_xticks(ctx_lengths)
    ax2.grid(True, alpha=0.25, axis="y")
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=11, loc="best")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_detail_tables(
    pdf: PdfPages,
    df: pd.DataFrame,
    models: list[str],
    ctx_lengths: list[int],
):
    """Per context-length tables with per-dataset results for every model."""
    model_labels = [get_display_name(m) for m in models]
    n_model = len(models)

    for ctx in ctx_lengths:
        sub = df[df["context_length"] == ctx]
        if sub.empty:
            continue

        for metric_suffix, metric_name in [
            ("mean_mae", "MAE"),
            ("mean_mse", "MSE"),
        ]:
            col_labels = ["Dataset", "N"] + model_labels

            rows = []
            agg_vals: dict[str, list[float]] = {m: [] for m in models}
            for _, row in sub.iterrows():
                ds = str(row["dataset"]).replace("_with_text", "")
                n = int(row["n_samples"]) if pd.notna(row.get("n_samples")) else 0
                r: list = [ds, str(n)]
                for m in models:
                    c = f"{m}_{metric_suffix}"
                    v = (
                        row[c]
                        if c in row.index and pd.notna(row[c])
                        else np.nan
                    )
                    r.append(v if not np.isnan(v) else "-")
                    agg_vals[m].append(v)
                rows.append(r)

            # Mean row
            mean_row: list = ["Mean", ""]
            for m in models:
                vals = [v for v in agg_vals[m] if not np.isnan(v)]
                mean_row.append(np.mean(vals) if vals else "-")
            rows.append(mean_row)

            ds_w = 0.12
            n_w = 0.03
            metric_w = (1.0 - ds_w - n_w) / n_model
            col_widths = [ds_w, n_w] + [metric_w] * n_model
            highlight = [list(range(2, 2 + n_model))]

            n_rows = len(rows)
            fig_h = max(8, n_rows * 0.35 + 3)
            fig, ax = plt.subplots(figsize=(18, fig_h))

            row_colors = [ALT_ROW_COLORS[i % 2] for i in range(n_rows - 1)] + [
                "#e3f2fd"
            ]

            _render_table(
                ax,
                col_labels,
                rows,
                f"Context Length {ctx} — {metric_name}",
                row_colors=row_colors,
                col_widths=col_widths,
                fontsize=8,
                highlight_groups=highlight,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


# ── Public API ───────────────────────────────────────────────────────────────


def run(
    output_dir: str | Path,
    summaries_dir: str | Path | None = None,
    out_path: str | Path | None = None,
    models: list[str] | None = None,
) -> bool:
    """Generate an aggregate PDF from evaluation results.

    Args:
        output_dir: Directory containing context_<N>/predictions/... layout.
        summaries_dir: Optional directory with LLM summary JSONs for quality
            filtering.  When None, all windows are included.
        out_path: Output PDF path.  Defaults to ``<output_dir>/aggregate_summary.pdf``.
        models: Model keys to include.  Defaults to MODEL_ORDER.

    Returns True on success.
    """
    if not HAS_MPL:
        print("matplotlib is required for aggregate PDF.  Install with: pip install matplotlib")
        return False

    output_dir = Path(output_dir)
    if out_path is None:
        out_path = output_dir / "aggregate_summary.pdf"
    else:
        out_path = Path(out_path)

    summaries_str = str(summaries_dir) if summaries_dir else None

    print(f"Loading prediction arrays from {output_dir} ...")
    try:
        df, stats_df = load_all_data(output_dir, summaries_str, models)
    except RuntimeError as e:
        print(f"  {e}")
        return False

    avail = available_models(df, models)
    if not avail:
        print("No models with data found.")
        return False

    ctx_lengths = sorted(df["context_length"].unique().tolist())

    if stats_df is not None and not stats_df.empty:
        first_ctx = stats_df["context_length"].min()
        dedup = stats_df[stats_df["context_length"] == first_ctx]
        total_w = int(dedup["n_total"].sum())
        bad_w = int(dedup["n_bad"].sum())
        print(
            f"  {len(df)} dataset-context rows, {total_w} total windows, "
            f"{bad_w} bad summaries excluded ({bad_w / total_w * 100:.1f}%)"
        )

    print(f"  Models: {', '.join(get_display_name(m) for m in avail)}")
    print(f"  Context lengths: {ctx_lengths}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(out_path)) as pdf:
        if stats_df is not None and not stats_df.empty:
            print("  Page: summary quality stats ...")
            page_summary_quality(pdf, stats_df)

        print("  Pages: aggregate metrics tables ...")
        page_aggregate_metrics(pdf, df, avail, ctx_lengths)

        print("  Page: win counts ...")
        page_win_counts(pdf, df, avail, ctx_lengths)

        print("  Page: Elo ratings ...")
        page_elo(pdf, df, avail, ctx_lengths)

        print("  Pages: per-context detail tables ...")
        page_detail_tables(pdf, df, avail, ctx_lengths)

    print(f"Aggregate PDF saved to {out_path}")
    return True


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate aggregate PDF from evaluation results across context lengths",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing context_<N>/predictions/... (the --output_dir from evaluation)",
    )
    parser.add_argument(
        "--summaries_dir",
        type=str,
        default=None,
        help="Directory with LLM summary JSONs for quality filtering (optional)",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Output PDF path (default: <output_dir>/aggregate_summary.pdf)",
    )
    args = parser.parse_args()

    ok = run(
        output_dir=args.output_dir,
        summaries_dir=args.summaries_dir,
        out_path=args.out_path,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
