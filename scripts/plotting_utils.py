"""
Shared forecasting plot utilities for Migas-1.5 and baselines (e.g. Chronos).

Use from scripts or from Jupyter notebooks (add repo root to path or install package).
Design: ground truth = black, Migas-1.5 = blue, Chronos = orange, forecast region emphasized (shaded + vline).
Metrics shown in titles: MAPE (%) and Error (MAE over horizon).

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

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Design: ground truth black; Chronos = saturated orange, Migas-1.5 = blue; historical dark gray
COLORS = {
    "ground_truth": "black",
    "historical": "#2C3E50",
    "Migas-1.5": "#E67E22",
    "Chronos": "#2980B9",
    "Chronos-2": "#2980B9",
}
# Fallback for any other model name
DEFAULT_MODEL_COLOR = "#3498DB"


def compute_mape(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean absolute percentage error (0-100 scale). gt and pred same shape, 1d or 2d."""
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    denom = np.abs(gt) + eps
    return float(np.mean(np.abs(pred - gt) / denom) * 100.0)


def compute_mae(gt: np.ndarray, pred: np.ndarray) -> float:
    """Mean absolute error over the array."""
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(gt))))


def _mape_mae_1d(gt_1d: np.ndarray, pred_1d: np.ndarray, eps: float = 1e-8) -> tuple[float, float]:
    """MAPE (%) and MAE for one sample (1d arrays of length pred_len)."""
    mape = float(np.mean(np.abs(pred_1d - gt_1d) / (np.abs(gt_1d) + eps)) * 100.0)
    mae = float(np.mean(np.abs(pred_1d - gt_1d)))
    return mape, mae


def _draw_forecast_region(ax: Any, context_len: int, pred_len: int) -> None:
    """Shaded forecast region and vertical line at context end."""
    ax.axvspan(
        context_len - 0.5, context_len + pred_len - 0.5, alpha=0.15, color="gray"
    )
    ax.axvline(
        x=context_len - 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7
    )


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
    xlabel: str = "Time Step",
    ylabel: str = "Value",
) -> None:
    """
    Draw one forecast window on the given axes.

    Args:
        ax: matplotlib axes
        history: shape (context_len,) — historical context
        gt: shape (pred_len,) — ground truth, or None for inference-only (no GT line, no metrics)
        preds: dict mapping model name -> array of shape (pred_len,). Keys like "Migas-1.5", "Chronos" get standard colors.
        context_len: length of history
        pred_len: length of gt and each prediction
        history_mean, history_std: if provided, history/gt/preds are treated as normalized and denormalized for plotting
        show_metrics: if True and gt is not None, add MAPE and Error to title for each model
        title: optional title when no metrics (e.g. dataset name)
        xlabel, ylabel: axis labels
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
        preds = {k: np.asarray(v, dtype=np.float64).ravel() * sigma + mu for k, v in preds.items()}
    else:
        preds = {k: np.asarray(v, dtype=np.float64).ravel() for k, v in preds.items()}

    t_input = np.arange(context_len)
    t_pred_extended = np.arange(context_len - 1, context_len + pred_len)
    last_input = float(hist[-1])

    _draw_forecast_region(ax, context_len, pred_len)

    ax.plot(t_input, hist, color=COLORS["historical"], linewidth=2.5, label="Historical", zorder=3)
    if has_gt:
        gt_extended = np.concatenate([[last_input], gt_arr])
        ax.plot(t_pred_extended, gt_extended, color=COLORS["ground_truth"], linewidth=2.5, label="Ground Truth", zorder=4)

    for model_name, pred_arr in preds.items():
        color = COLORS.get(model_name, DEFAULT_MODEL_COLOR)
        pred_extended = np.concatenate([[last_input], pred_arr])
        ax.plot(
            t_pred_extended,
            pred_extended,
            color=color,
            linewidth=2.5,
            label=model_name,
            zorder=5,
            alpha=0.9,
        )

    ax.set_xlim(-0.5, context_len + pred_len - 0.5)
    ax.set_xticks(np.arange(0, context_len + pred_len, max(1, (context_len + pred_len) // 10)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    if show_metrics and preds and has_gt and gt_arr is not None:
        parts = []
        for model_name, pred_arr in preds.items():
            mape, mae = _mape_mae_1d(gt_arr, pred_arr)
            parts.append(f"{model_name} MAPE={mape:.2f}% Err={mae:.4f}")
        ax.set_title("\n".join(parts), fontsize=9)
    elif title:
        ax.set_title(title, fontsize=9)


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
    figsize: tuple[float, float] = (7.0, 4.0),
    show_metrics: bool = True,
) -> tuple[plt.Figure, Any]:
    """
    One figure with one window: history + optional ground truth + Migas-1.5 and/or Chronos.

    Args:
        history: (context_len,)
        gt: (pred_len,) or None for inference-only (no GT line, no metrics)
        preds: {"Migas-1.5": (pred_len,), "Chronos": (pred_len,)} or just one key
        context_len, pred_len: lengths
        history_mean, history_std: optional denormalization
        title: optional subplot title (e.g. dataset name)
        figsize, show_metrics: passed through

    Returns:
        (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)
    plot_one_forecast(
        ax, history, gt, preds, context_len, pred_len,
        history_mean=history_mean, history_std=history_std,
        show_metrics=show_metrics, title=title,
    )
    plt.tight_layout()
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
    figsize_per_subplot: tuple[float, float] = (5.5, 3.0),
    max_cols: int = 3,
    show_metrics: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Figure with a grid of subplots, one per sample index.

    Args:
        history_2d: (n_samples, context_len)
        gt_2d: (n_samples, pred_len), or None for inference-only (no GT line, no metrics)
        preds_2d: {"Migas-1.5": (n_samples, pred_len), "Chronos": (n_samples, pred_len)}
        context_len, pred_len: lengths
        sample_indices: which rows to plot (e.g. [0, 1, 2, 5])
        history_means, history_stds: optional (n_samples,) for denormalization
        titles: optional list of title strings, one per sample (e.g. ["Sample 0", "Sample 1"])
        figsize_per_subplot, max_cols: layout
        show_metrics: include MAPE and Error in each subplot title

    Returns:
        (fig, axes) with axes 2d array (n_rows, n_cols)
    """
    n_plots = len(sample_indices)
    n_cols = min(max_cols, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
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

        subplot_title = (titles[idx] if titles and idx < len(titles) else None) or f"Sample {sample_idx}"

        plot_one_forecast(
            ax, hist, gt, preds_one, context_len, pred_len,
            history_mean=mu, history_std=sigma,
            show_metrics=show_metrics, title=subplot_title,
        )

    for idx in range(n_plots, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    return fig, axes
