"""Scenario-specific plotting helpers for counterfactual analysis.

Integrates with the shared ``scripts.plotting_utils`` style and color palette
and adds overlays and comparison views tailored to counterfactual experiments.
"""

from __future__ import annotations

from typing import Any, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.plotting_utils import COLORS, _draw_forecast_region, apply_migas_style

    apply_migas_style()
except ImportError:
    COLORS = {
        "ground_truth": "#1B2631",
        "historical": "#5D6D7E",
        "Migas-1.5": "#F28C28",
        "forecast_region": "#FDF6EC",
        "forecast_vline": "#ABB2B9",
    }

    def _draw_forecast_region(
        ax: Any, context_len: int, pred_len: int, **kwargs
    ) -> None:
        left = kwargs.get("boundary", context_len - 0.5)
        right = kwargs.get("boundary_end", context_len + pred_len - 0.5)
        ax.axvspan(left, right, alpha=0.45, color="#FDF6EC", zorder=0)
        ax.axvline(
            x=left, color="#ABB2B9", linestyle="--", linewidth=0.9, alpha=0.6, zorder=1
        )

    def apply_migas_style() -> None:
        pass


CF_COLOR = "#C0392B"
CF_COLOR_FILL = "#FADBD8"


def plot_scenario_comparison(
    context: np.ndarray,
    original_forecast: np.ndarray,
    counterfactual_forecast: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    *,
    direction: str = "up",
    title: str | None = None,
    slope_shift: float | None = None,
    trend_delta: float | None = None,
    figsize: tuple[float, float] = (9, 4.5),
    ax: Any | None = None,
    timestamps=None,
) -> tuple[plt.Figure | None, Any]:
    """Overlay original and counterfactual forecasts on a single axes."""
    import pandas as pd

    context = np.asarray(context).ravel()
    original_forecast = np.asarray(original_forecast).ravel()
    counterfactual_forecast = np.asarray(counterfactual_forecast).ravel()

    ctx_len = len(context)
    pred_len = len(original_forecast)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    use_dates = timestamps is not None and len(timestamps) == ctx_len + pred_len
    if use_dates:
        ts = pd.to_datetime(list(timestamps))
        t_ctx = ts[:ctx_len]
        t_pred = pd.DatetimeIndex([ts[ctx_len - 1]] + list(ts[ctx_len:]))
        _draw_forecast_region(
            ax, ctx_len, pred_len, boundary=t_pred[0], boundary_end=t_pred[-1]
        )
    else:
        t_ctx = np.arange(ctx_len)
        t_pred = np.arange(ctx_len - 1, ctx_len + pred_len)
        _draw_forecast_region(ax, ctx_len, pred_len)

    last_val = float(context[-1])

    ax.plot(
        t_ctx,
        context,
        color=COLORS["historical"],
        lw=2.0,
        label="Historical",
        zorder=3,
        solid_capstyle="round",
    )

    if ground_truth is not None:
        gt = np.asarray(ground_truth).ravel()
        gt_ext = np.concatenate([[last_val], gt])
        ax.plot(
            t_pred,
            gt_ext,
            color=COLORS["ground_truth"],
            lw=2.2,
            label="Ground Truth",
            zorder=4,
            solid_capstyle="round",
        )

    orig_ext = np.concatenate([[last_val], original_forecast])
    ax.plot(
        t_pred,
        orig_ext,
        color=COLORS["Migas-1.5"],
        lw=2.0,
        ls="--",
        label="Migas-1.5 (original)",
        zorder=5,
        alpha=0.85,
        solid_capstyle="round",
    )

    cf_ext = np.concatenate([[last_val], counterfactual_forecast])
    direction_label = "bullish" if direction == "up" else "bearish"
    ax.plot(
        t_pred,
        cf_ext,
        color=CF_COLOR,
        lw=2.4,
        label=f"Migas-1.5 ({direction_label} scenario)",
        zorder=6,
        solid_capstyle="round",
    )

    # Shaded area between original and counterfactual
    ax.fill_between(
        t_pred,
        orig_ext,
        cf_ext,
        color=CF_COLOR_FILL,
        alpha=0.35,
        zorder=2,
    )

    if use_dates:
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
        )
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(35)
            lbl.set_ha("right")
        ax.set_xlabel("Date", color="#566573")
    else:
        ax.set_xlim(-0.5, ctx_len + pred_len - 0.5)
        ax.set_xlabel("Time Step", color="#566573")
    ax.set_ylabel("Value", color="#566573")
    ax.legend(
        loc="best",
        fontsize=8,
        handlelength=1.6,
        labelspacing=0.35,
        borderpad=0.45,
    )

    if title:
        ax.set_title(title, fontweight="bold", fontsize=11)
    elif slope_shift is not None and trend_delta is not None:
        badge = f"Slope shift {slope_shift:+.4f}   ·   Trend delta {trend_delta:+.3f}"
        ax.annotate(
            badge,
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            fontsize=8.5,
            color="#5D6D7E",
            ha="center",
            va="bottom",
            xytext=(0, 6),
            textcoords="offset points",
        )

    if fig is not None:
        fig.tight_layout(pad=1.2)
    return fig, ax


def plot_scenario_grid(
    results,  # list[ScenarioResult]
    *,
    max_cols: int = 3,
    figsize_per: tuple[float, float] = (6, 3.5),
    suptitle: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Grid of scenario comparison subplots, one per :class:`ScenarioResult`."""
    n = len(results)
    n_cols = min(max_cols, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per[0] * n_cols, figsize_per[1] * n_rows),
        squeeze=False,
    )

    for idx, r in enumerate(results):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        plot_scenario_comparison(
            r.context,
            r.original_forecast,
            r.counterfactual_forecast,
            ground_truth=r.ground_truth,
            direction=r.direction,
            slope_shift=r.slope_shift,
            trend_delta=r.trend_delta,
            ax=ax,
        )
        badge = f"Window {idx}  ·  slope shift {r.slope_shift:+.4f}"
        ax.annotate(
            badge,
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            fontsize=7.5,
            color="#5D6D7E",
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02, color="#1B2631")
    fig.tight_layout(pad=1.0, h_pad=2.0, w_pad=1.5)
    return fig, axes


def plot_trend_summary(
    results,  # list[ScenarioResult]
    *,
    figsize: tuple[float, float] = (9, 4),
) -> tuple[plt.Figure, Any]:
    """Bar chart summarizing trend deltas across windows."""
    deltas = [r.trend_delta for r in results]
    slopes = [r.slope_shift for r in results]
    x = np.arange(len(results))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    bar_pos = [COLORS["Migas-1.5"] if d > 0 else "#6C8EBF" for d in deltas]
    bars1 = ax1.bar(
        x, deltas, color=bar_pos, alpha=0.85, edgecolor="white", lw=0.6, zorder=3
    )
    ax1.axhline(0, color="#ABB2B9", lw=0.7, zorder=2)
    ax1.set_xlabel("Window", color="#566573")
    ax1.set_ylabel("Trend Score Delta", color="#566573")
    ax1.set_title("Trend Score Change", fontsize=11, fontweight=600)

    bar_pos2 = [COLORS["Migas-1.5"] if s > 0 else "#6C8EBF" for s in slopes]
    bars2 = ax2.bar(
        x, slopes, color=bar_pos2, alpha=0.85, edgecolor="white", lw=0.6, zorder=3
    )
    ax2.axhline(0, color="#ABB2B9", lw=0.7, zorder=2)
    ax2.set_xlabel("Window", color="#566573")
    ax2.set_ylabel("Slope Shift", color="#566573")
    ax2.set_title("Forecast Slope Shift", fontsize=11, fontweight=600)

    fig.tight_layout(pad=1.2)
    return fig, (ax1, ax2)


def display_text_comparison(
    original_summary: str,
    counterfactual_summary: str,
) -> str:
    """Return an HTML string highlighting the original vs. counterfactual
    summaries for rendering in a Jupyter notebook via ``IPython.display.HTML``.
    """
    teal = COLORS["Migas-1.5"]
    return f"""
    <style>
      .migas-text-compare {{
        display: flex;
        gap: 20px;
        font-family: system-ui, -apple-system, sans-serif;
        font-size: 13px;
      }}
      .migas-text-card {{
        flex: 1;
        padding: 16px;
        border-radius: 8px;
        line-height: 1.5;
        color: #111827;
      }}
      .migas-text-card-original {{
        background: #F8FFFE;
        border-left: 4px solid {teal};
      }}
      .migas-text-card-counterfactual {{
        background: #FDF2F0;
        border-left: 4px solid {CF_COLOR};
      }}
      @media (prefers-color-scheme: dark) {{
        .migas-text-card {{
          color: #E5E7EB;
        }}
        .migas-text-card-original {{
          background: #10211D;
        }}
        .migas-text-card-counterfactual {{
          background: #2B1714;
        }}
      }}
    </style>
    <div class="migas-text-compare">
      <div class="migas-text-card migas-text-card-original">
        <b style="color:{teal}">Original Summary</b><br><br>{_html_escape(original_summary)}
      </div>
      <div class="migas-text-card migas-text-card-counterfactual">
        <b style="color:{CF_COLOR}">Counterfactual Summary</b><br><br>{_html_escape(counterfactual_summary)}
      </div>
    </div>
    """


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
