"""
Shared forecasting plot utilities for Migas-1.5 and baselines (e.g. Chronos).

Use from scripts or from Jupyter notebooks (add repo root to path or install package).

Call ``apply_migas_style()`` once at import time (done automatically) to set up
consistent rcParams across all notebooks and scripts.

Color scheme:
  - Historical context = slate gray
  - Ground truth = near-black charcoal
  - Migas-1.5 = vibrant teal (hero color)
  - Chronos / Chronos-2 = steel blue
  - Forecast region = subtle warm wash

Examples:
  # Single window, Migas-1.5 only
  from scripts.plotting_utils import plot_forecast_single, COLORS
  fig, ax = plot_forecast_single(history, gt, {"Migas-1.5": migas_pred}, context_len, pred_len)

  # Single window, Migas-1.5 and Chronos
  plot_forecast_single(history, gt, {"Migas-1.5": migas_pred, "Chronos": chronos_pred}, context_len, pred_len)

  # Grid of windows (multiple subplots)
  plot_forecast_grid(history_2d, gt_2d, {"Migas-1.5": migas_2d, "Chronos": chronos_2d},
                     context_len, pred_len, sample_indices=[0, 1, 2])
"""

from __future__ import annotations

import textwrap
from typing import Any, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Brand palette
# ---------------------------------------------------------------------------
COLORS = {
    "ground_truth": "#1B2631",
    "historical": "#5D6D7E",
    "Migas-1.5": "#F28C28",
    "Chronos": "#4A90D9",
    "Chronos-2": "#4A90D9",
    "TimesFM": "#7B68EE",
    "forecast_region": "#FDF6EC",
    "forecast_vline": "#ABB2B9",
}
DEFAULT_MODEL_COLOR = "#6C8EBF"

_STYLE_APPLIED = False


def apply_migas_style() -> None:
    """Set matplotlib rcParams for a clean, modern, release-quality look.

    Safe to call multiple times — only applies once per process.
    """
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    _STYLE_APPLIED = True

    plt.rcParams.update(
        {
            # --- typography ---
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Inter",
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
                "Liberation Sans",
                "sans-serif",
            ],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": 600,
            "axes.labelsize": 11,
            "axes.labelweight": 500,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.title_fontsize": 10,
            # --- figure ---
            "figure.dpi": 140,
            "figure.facecolor": "white",
            "figure.edgecolor": "none",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
            "savefig.facecolor": "white",
            # --- axes ---
            "axes.facecolor": "#FAFBFC",
            "axes.edgecolor": "#D5D8DC",
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            # --- grid ---
            "grid.color": "#E5E8E8",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.8,
            "grid.linestyle": "-",
            # --- ticks ---
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,
            "xtick.color": "#566573",
            "ytick.color": "#566573",
            # --- lines ---
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "lines.antialiased": True,
            # --- legend ---
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#D5D8DC",
            "legend.fancybox": True,
            "legend.borderpad": 0.5,
            "legend.columnspacing": 1.2,
            "legend.handlelength": 1.8,
            # --- layout ---
            "figure.constrained_layout.use": False,
        }
    )


apply_migas_style()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_mape(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean absolute percentage error (0-100 scale). gt and pred same shape, 1d or 2d."""
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    denom = np.abs(gt) + eps
    return float(np.mean(np.abs(pred - gt) / denom) * 100.0)


def compute_mae(gt: np.ndarray, pred: np.ndarray) -> float:
    """Mean absolute error over the array."""
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(gt))))


def _mape_mae_1d(
    gt_1d: np.ndarray, pred_1d: np.ndarray, eps: float = 1e-8
) -> tuple[float, float]:
    """MAPE (%) and MAE for one sample (1d arrays of length pred_len)."""
    mape = float(np.mean(np.abs(pred_1d - gt_1d) / (np.abs(gt_1d) + eps)) * 100.0)
    mae = float(np.mean(np.abs(pred_1d - gt_1d)))
    return mape, mae


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _draw_forecast_region(
    ax: Any,
    context_len: int,
    pred_len: int,
    *,
    boundary: Any | None = None,
    boundary_end: Any | None = None,
) -> None:
    """Warm-tinted forecast band with a subtle divider line.

    When *boundary* / *boundary_end* are given (e.g. datetime objects) they are
    used instead of the integer ``context_len - 0.5`` / ``context_len + pred_len - 0.5``.
    """
    left = boundary if boundary is not None else context_len - 0.5
    right = boundary_end if boundary_end is not None else context_len + pred_len - 0.5

    ax.axvspan(left, right, alpha=0.45, color=COLORS["forecast_region"], zorder=0)
    ax.axvline(
        x=left,
        color=COLORS["forecast_vline"],
        linestyle="--",
        linewidth=0.9,
        alpha=0.6,
        zorder=1,
    )


def _format_metric_badge(model_name: str, mape: float, mae: float) -> str:
    """Compact metric string for annotation boxes."""
    return f"{model_name}  MAPE {mape:.2f}%  ·  MAE {mae:.4f}"


# ---------------------------------------------------------------------------
# Core single-axes plot
# ---------------------------------------------------------------------------


def plot_one_forecast(
    ax: Any,
    history: np.ndarray,
    gt: np.ndarray | None,
    preds: dict[str, np.ndarray],
    context_len: int,
    pred_len: int,
    *,
    history_mean: float | None = None,
    history_std: float | None = None,
    show_metrics: bool = True,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Value",
    timestamps: Sequence | np.ndarray | None = None,
) -> None:
    """
    Draw one forecast window on the given axes.

    Args:
        ax: matplotlib axes
        history: shape (context_len,) — historical context
        gt: shape (pred_len,) — ground truth, or None for inference-only
        preds: dict mapping model name -> array of shape (pred_len,)
        context_len: length of history
        pred_len: length of gt and each prediction
        history_mean, history_std: optional denormalization params
        show_metrics: if True and gt is not None, add MAPE/MAE badge
        title: optional title (e.g. dataset name)
        xlabel: x-axis label (auto-set to "Date" or "Time Step" based on *timestamps*)
        ylabel: y-axis label
        timestamps: optional sequence of length ``context_len + pred_len`` with
            datetime-like objects.  When provided the x-axis shows dates instead
            of integer step indices.
    """
    hist = np.asarray(history, dtype=np.float64).ravel()
    if len(hist) != context_len:
        raise ValueError(f"history len {len(hist)} != context_len {context_len}")
    has_gt = gt is not None
    if has_gt:
        gt_arr = np.asarray(gt, dtype=np.float64).ravel()
        if len(gt_arr) != pred_len:
            raise ValueError(f"gt len {len(gt_arr)} != pred_len {pred_len}")
    else:
        gt_arr = None

    if history_mean is not None and history_std is not None:
        mu, sigma = float(history_mean), float(history_std)
        hist = hist * sigma + mu
        if has_gt:
            gt_arr = gt_arr * sigma + mu
        preds = {
            k: np.asarray(v, dtype=np.float64).ravel() * sigma + mu
            for k, v in preds.items()
        }
    else:
        preds = {k: np.asarray(v, dtype=np.float64).ravel() for k, v in preds.items()}

    last_input = float(hist[-1])

    # ---- resolve x-axis values ----
    use_dates = timestamps is not None and len(timestamps) == context_len + pred_len
    if use_dates:
        import pandas as pd

        ts = pd.to_datetime(list(timestamps))
        t_input = ts[:context_len]
        t_pred_extended = pd.DatetimeIndex(
            [ts[context_len - 1]] + list(ts[context_len:])
        )
        _draw_forecast_region(
            ax,
            context_len,
            pred_len,
            boundary=ts[context_len - 1],
            boundary_end=ts[-1],
        )
    else:
        t_input = np.arange(context_len)
        t_pred_extended = np.arange(context_len - 1, context_len + pred_len)
        _draw_forecast_region(ax, context_len, pred_len)

    # Historical context
    ax.plot(
        t_input,
        hist,
        color=COLORS["historical"],
        linewidth=2.0,
        label="Historical",
        zorder=3,
        solid_capstyle="round",
    )

    # Ground truth
    if has_gt:
        gt_extended = np.concatenate([[last_input], gt_arr])
        ax.plot(
            t_pred_extended,
            gt_extended,
            color=COLORS["ground_truth"],
            linewidth=2.2,
            label="Ground Truth",
            zorder=4,
            solid_capstyle="round",
        )

    # Draw Migas last so it renders on top
    def _model_order(item: tuple[str, np.ndarray]) -> int:
        return 1 if "Migas" in item[0] else 0

    for model_name, pred_arr in sorted(preds.items(), key=_model_order):
        color = COLORS.get(model_name, DEFAULT_MODEL_COLOR)
        pred_extended = np.concatenate([[last_input], pred_arr])
        ax.plot(
            t_pred_extended,
            pred_extended,
            color=color,
            linewidth=2.4,
            label=model_name,
            zorder=5,
            alpha=0.92,
            solid_capstyle="round",
        )

    # ---- x-axis formatting ----
    if use_dates:
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
        )
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(35)
            lbl.set_ha("right")
    else:
        ax.set_xlim(-0.5, context_len + pred_len - 0.5)
        step = max(1, (context_len + pred_len) // 8)
        ax.set_xticks(np.arange(0, context_len + pred_len, step))

    if xlabel is None:
        xlabel = "Date" if use_dates else "Time Step"
    ax.set_xlabel(xlabel, color="#566573")
    ax.set_ylabel(ylabel, color="#566573")

    # Legend
    ax.legend(
        loc="best",
        fontsize=8,
        handlelength=1.6,
        labelspacing=0.35,
        borderpad=0.45,
    )

    # Title / metrics badge
    if show_metrics and preds and has_gt and gt_arr is not None:
        parts = []
        for model_name, pred_arr in preds.items():
            mape, mae = _mape_mae_1d(gt_arr, pred_arr)
            parts.append(_format_metric_badge(model_name, mape, mae))
        badge = "    |    ".join(parts)
        ax.annotate(
            badge,
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            fontsize=8,
            color="#5D6D7E",
            ha="center",
            va="bottom",
            xytext=(0, 6),
            textcoords="offset points",
        )
        if title:
            ax.set_title(title, pad=18)
    elif title:
        ax.set_title(title)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def _format_summary_text(text: str, width: int) -> str:
    """Format summary text with styled section headers and wrapped body text.

    Sections starting with 'HEADER:' are reformatted as:
        ◆ HEADER
        ─────────────────────
        <wrapped body text>
    """
    _RULE_CHAR = "─"
    paragraphs = text.strip().split("\n\n")
    parts = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Detect a section header: first line ends with ':'
        first_line, _, rest = para.partition("\n")
        first_line = first_line.strip()
        if first_line.endswith(":") and len(first_line) < 60:
            header = first_line.rstrip(":")
            body = (
                " ".join(rest.split()).strip()
                if rest.strip()
                else " ".join(para.split()[len(first_line.split()) :]).strip()
            )
            rule = _RULE_CHAR * min(width, 52)
            header_block = f"• {header}\n{rule}"
            if body:
                wrapped_body = textwrap.fill(body, width=width)
                parts.append(f"{header_block}\n{wrapped_body}")
            else:
                parts.append(header_block)
        else:
            body = " ".join(para.split()).strip()
            parts.append(textwrap.fill(body, width=width))
    return "\n\n".join(parts)


def plot_forecast_single(
    history: np.ndarray,
    gt: np.ndarray | None,
    preds: dict[str, np.ndarray],
    context_len: int,
    pred_len: int,
    *,
    history_mean: float | None = None,
    history_std: float | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8.0, 4.0),
    show_metrics: bool = True,
    timestamps: Sequence | np.ndarray | None = None,
    text_summary: str | None = None,
    summary_fontsize: int = 8,
) -> tuple[plt.Figure, Any]:
    """
    One figure with one window: history + optional ground truth + model forecasts.

    Args:
        timestamps: optional sequence of length ``context_len + pred_len`` with
            datetime-like objects.  When provided the x-axis shows dates.
        text_summary: optional summary string printed below the plot in a styled
            text panel.  Paragraph breaks (FACTUAL / PREDICTIVE sections) are
            preserved; long lines are wrapped automatically.
        summary_fontsize: font size for the summary text panel (default 8).

    Returns:
        (fig, ax)
    """
    if text_summary is None:
        fig, ax = plt.subplots(figsize=figsize)
        plot_one_forecast(
            ax,
            history,
            gt,
            preds,
            context_len,
            pred_len,
            history_mean=history_mean,
            history_std=history_std,
            show_metrics=show_metrics,
            title=title,
            timestamps=timestamps,
        )
        fig.tight_layout(pad=1.2)
        return fig, ax

    # ---- layout with text panel below ----
    # Wrap text and estimate the height it needs in inches.
    wrap_width = max(60, int(figsize[0] * 11))
    wrapped = _format_summary_text(text_summary, width=wrap_width)
    n_lines = wrapped.count("\n") + 1
    pts_per_line = summary_fontsize * 1.55  # leading
    text_height_in = n_lines * pts_per_line / 72 + 0.55  # + padding

    fig = plt.figure(
        figsize=(figsize[0], figsize[1] + text_height_in), layout="constrained"
    )
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[figsize[1], text_height_in],
        hspace=0.35,
    )
    ax = fig.add_subplot(gs[0])
    text_ax = fig.add_subplot(gs[1])

    plot_one_forecast(
        ax,
        history,
        gt,
        preds,
        context_len,
        pred_len,
        history_mean=history_mean,
        history_std=history_std,
        show_metrics=show_metrics,
        title=title,
        timestamps=timestamps,
    )

    text_ax.axis("off")
    text_ax.text(
        0.5,
        1.0,
        wrapped,
        ha="center",
        va="top",
        fontsize=summary_fontsize,
        color="#2C3E50",
        transform=text_ax.transAxes,
        linespacing=1.6,
        bbox=dict(
            boxstyle="round,pad=0.7",
            facecolor="#F4F6F7",
            edgecolor="#D5D8DC",
            alpha=0.9,
        ),
        fontfamily="sans-serif",
    )

    return fig, ax


def plot_forecast_grid(
    history_2d: np.ndarray,
    gt_2d: np.ndarray | None,
    preds_2d: dict[str, np.ndarray],
    context_len: int,
    pred_len: int,
    sample_indices: list[int],
    *,
    history_means: np.ndarray | None = None,
    history_stds: np.ndarray | None = None,
    titles: list[str] | None = None,
    figsize_per_subplot: tuple[float, float] = (5.5, 3.2),
    max_cols: int = 3,
    show_metrics: bool = True,
    timestamps_2d: Sequence[Sequence] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Figure with a grid of subplots, one per sample index.

    Args:
        history_2d: (n_samples, context_len)
        gt_2d: (n_samples, pred_len), or None for inference-only
        preds_2d: {"Migas-1.5": (n_samples, pred_len), ...}
        context_len, pred_len: lengths
        sample_indices: which rows to plot
        history_means, history_stds: optional (n_samples,) for denorm
        titles: optional list of title strings
        figsize_per_subplot, max_cols: layout
        show_metrics: include MAPE and Error badge
        timestamps_2d: optional list of timestamp sequences, one per sample
            in *sample_indices* order (not indexed by sample_idx).  Each
            sequence should have length ``context_len + pred_len``.

    Returns:
        (fig, axes) with axes 2d array (n_rows, n_cols)
    """
    n_plots = len(sample_indices)
    n_cols = min(max_cols, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows),
        squeeze=False,
    )

    for idx, sample_idx in enumerate(sample_indices):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        hist = history_2d[sample_idx]
        gt = gt_2d[sample_idx] if gt_2d is not None else None
        preds_one = {k: v[sample_idx] for k, v in preds_2d.items()}

        mu = float(history_means[sample_idx]) if history_means is not None else None
        sigma = float(history_stds[sample_idx]) if history_stds is not None else None

        subplot_title = (
            titles[idx] if titles and idx < len(titles) else None
        ) or f"Sample {sample_idx}"

        ts = timestamps_2d[idx] if timestamps_2d is not None else None

        plot_one_forecast(
            ax,
            hist,
            gt,
            preds_one,
            context_len,
            pred_len,
            history_mean=mu,
            history_std=sigma,
            show_metrics=show_metrics,
            title=subplot_title,
            timestamps=ts,
        )

    for idx in range(n_plots, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    fig.tight_layout(pad=1.0, h_pad=2.0, w_pad=1.5)
    return fig, axes
