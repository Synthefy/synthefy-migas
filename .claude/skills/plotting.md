# Plotting Migas-1.5 Forecasts

When the user asks you to plot, visualize, or chart forecasts, use the plotting
utilities from `migaseval.plotting_utils`. Always call `apply_migas_style()` once
before any plotting to get consistent, release-quality styling.

## Setup

```python
from migaseval.plotting_utils import (
    COLORS,
    apply_migas_style,
    plot_forecast_single,
    plot_forecast_grid,
    _draw_forecast_region,
)

apply_migas_style()  # call once — sets rcParams for clean, modern look
```

## Color Palette

Use the `COLORS` dict for consistent styling across all plots:

| Key | Hex | Use |
|-----|-----|-----|
| `"historical"` | `#5D6D7E` | Context window (slate gray) |
| `"ground_truth"` | `#1B2631` | Actual future values (charcoal) |
| `"Migas-1.5"` | `#F28C28` | Migas-1.5 forecast (vibrant teal/orange — hero color) |
| `"Chronos-2"` | `#4A90D9` | Chronos baseline (steel blue) |
| `"forecast_region"` | `#FDF6EC` | Background shading for forecast zone |
| `"forecast_vline"` | `#ABB2B9` | Vertical divider at forecast boundary |

For counterfactual scenarios, the notebooks use:
- Bullish: `#2EAD6D` (green)
- Bearish: `#C0392B` (red)
- Scenario fill: `#9B8EC4` with `alpha=0.08`

## Single Forecast Plot

`plot_forecast_single` creates a one-window figure with optional ground truth,
metrics badge, and text summary panel below the chart.

```python
fig, ax = plot_forecast_single(
    history,                          # np.ndarray, shape (context_len,)
    gt_vals,                          # np.ndarray, shape (pred_len,) or None
    {"Chronos-2": chronos_fc, "Migas-1.5": migas_fc},  # dict of forecasts
    context_len,                      # int — SEQ_LEN
    pred_len,                         # int — PRED_LEN
    title="Chronos-2 vs. Migas-1.5",
    figsize=(11, 4),
    show_metrics=True,                # adds MAPE/MAE badge when gt is provided
    text_summary=summary,             # optional — renders summary below the plot
    timestamps=full["t"].values,      # optional — uses dates on x-axis instead of step indices
)
plt.show()
```

**Key behaviors:**
- When `timestamps` is provided (length = context_len + pred_len), the x-axis shows dates
- When `text_summary` is provided, a styled text panel is added below the chart
- When `show_metrics=True` and ground truth exists, a MAPE/MAE badge appears above the title
- Migas-1.5 is always drawn on top (highest z-order)

## Grid of Forecast Windows

`plot_forecast_grid` creates a multi-subplot figure for comparing multiple windows
(e.g. from a backtest or batch run).

```python
fig, axes = plot_forecast_grid(
    history_2d,                       # np.ndarray, shape (n_samples, context_len)
    gt_2d,                            # np.ndarray, shape (n_samples, pred_len) or None
    {"Migas-1.5": migas_2d, "Chronos-2": chronos_2d},  # dict of (n_samples, pred_len)
    context_len,
    pred_len,
    sample_indices=[0, 1, 2, 3],      # which rows to plot
    titles=["Window 1", "Window 2", "Window 3", "Window 4"],  # optional
    max_cols=3,                        # max columns in grid
    show_metrics=True,
)
plt.show()
```

## Forecast Region Shading

`_draw_forecast_region` adds a warm-tinted background band and vertical divider
to mark the forecast zone. Used internally by the plot functions, but can be
called directly for custom plots:

```python
_draw_forecast_region(
    ax, context_len, pred_len,
    boundary=t_pred[0],       # datetime or int — left edge of forecast zone
    boundary_end=t_pred[-1],  # datetime or int — right edge
)
```

## Counterfactual Scenario Fan (Custom Plot)

For the scenario fan plot (original + bullish + bearish), build it manually since
it needs custom colors and fill. Pattern from the notebooks:

```python
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(11, 5))
_draw_forecast_region(ax, CONTEXT_LEN, PRED_LEN, boundary=t_pred[0], boundary_end=t_pred[-1])

# Historical
ax.plot(t_ctx, context_vals, color=COLORS["historical"], lw=2.0, label="Historical")

# Ground truth (if available)
ax.plot(t_pred, np.concatenate([[last_val], gt_vals]),
        color=COLORS["ground_truth"], lw=2.2, label="Ground Truth")

# Original forecast
ax.plot(t_pred, np.concatenate([[last_val], fc_original]),
        color=COLORS["Migas-1.5"], lw=2.0, ls="--", alpha=0.85, label="Original (Migas-1.5)")

# Bullish / Bearish
ax.plot(t_pred, np.concatenate([[last_val], fc_bullish]),
        color="#2EAD6D", lw=2.4, label="Bullish scenario")
ax.plot(t_pred, np.concatenate([[last_val], fc_bearish]),
        color="#C0392B", lw=2.4, label="Bearish scenario")

# Scenario range fill
ax.fill_between(t_pred,
    np.concatenate([[last_val], fc_bearish]),
    np.concatenate([[last_val], fc_bullish]),
    alpha=0.08, color="#9B8EC4", label="Scenario range")

ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.legend(fontsize=8, handlelength=1.6)
fig.tight_layout(pad=1.2)
plt.show()
```

## Important Notes

- Always prepend `last_val` (the last context value) to forecast arrays when plotting
  prediction lines — this connects the forecast to the historical line visually:
  `np.concatenate([[last_val], forecast])`
- `t_pred` should have length `pred_len + 1` to match the prepended arrays
- For date x-axes, rotate labels: `lbl.set_rotation(35); lbl.set_ha("right")`
- Denormalization is built-in: pass `history_mean` and `history_std` to automatically
  rescale values back to original units
