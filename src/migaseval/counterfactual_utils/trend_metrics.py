"""Directional trend metrics for evaluating forecast steering.

Scoring functions that measure how strongly a forecast trends in a desired
direction.  Used as objective functions for counterfactual scenario selection
and for reporting in evaluation notebooks.

All core functions accept a 1-D numpy array of forecast values and return a
float.  Higher positive values indicate stronger alignment with the requested
direction.
"""

from __future__ import annotations

import numpy as np


def linear_slope(y: np.ndarray) -> float:
    """Slope of a least-squares linear fit through *y*.

    Positive = upward trend, negative = downward.  Scale depends on value
    magnitude; compare within the same series only.
    """
    x = np.arange(len(y), dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def endpoint_change(y: np.ndarray) -> float:
    """Relative change from first to last predicted value.

    Normalized by max(|y[0]|, std(y)) so the result stays bounded.
    """
    denom = max(abs(float(y[0])), float(np.std(y)), 1e-8)
    return float((y[-1] - y[0]) / denom)


def monotonicity(y: np.ndarray, direction: str = "up") -> float:
    """Fraction of consecutive step-pairs moving in *direction*.

    Returns a value in [0, 1].  1.0 = every step moves the right way.
    """
    diffs = np.diff(y)
    if direction == "up":
        return float(np.mean(diffs > 0))
    return float(np.mean(diffs < 0))


def breakout_ratio(
    y_forecast: np.ndarray,
    y_history: np.ndarray,
    direction: str = "up",
) -> float:
    """How far the forecast breaks beyond the historical range.

    For direction="up":   rewards forecasts above history max.
    For direction="down": rewards forecasts below history min.

    Returns >= 0 (0 = stays within bounds).  Normalized by history range.
    """
    h_range = float(np.ptp(y_history)) + 1e-8

    if direction == "down":
        h_min = float(np.min(y_history))
        penetration = h_min - y_forecast
    else:
        h_max = float(np.max(y_history))
        penetration = y_forecast - h_max

    mean_pen = float(np.mean(np.clip(penetration, 0, None)))
    max_pen = float(np.max(np.clip(penetration, 0, None)))
    return (0.5 * mean_pen + 0.5 * max_pen) / h_range


def exceedance_fraction(
    y_forecast: np.ndarray,
    y_history: np.ndarray,
    direction: str = "up",
) -> float:
    """Fraction of forecast timesteps beyond the historical extremum.

    Returns a value in [0, 1].
    """
    if direction == "down":
        threshold = float(np.min(y_history))
        return float(np.mean(y_forecast < threshold))
    threshold = float(np.max(y_history))
    return float(np.mean(y_forecast > threshold))


def composite_trend_score(
    y: np.ndarray,
    direction: str = "up",
    y_history: np.ndarray | None = None,
    *,
    w_slope: float = 0.25,
    w_endpoint: float = 0.15,
    w_monotonicity: float = 0.15,
    w_breakout: float = 0.30,
    w_exceedance: float = 0.15,
) -> float:
    """Weighted composite trend score.

    When *y_history* is provided the score heavily rewards forecasts that
    break beyond the historical range (breakout + exceedance = 45 % weight).
    Without it the weights are redistributed to slope and endpoint.

    Returns roughly in [-1, 2+]; positive = forecast trends the right way.
    """
    sign = 1.0 if direction == "up" else -1.0

    slope = linear_slope(y) * sign
    y_range = float(np.ptp(y)) + 1e-8
    norm_slope = float(np.clip(slope / y_range * len(y), -2, 2))

    ep = endpoint_change(y) * sign
    norm_ep = float(np.clip(ep, -2, 2))

    mono = monotonicity(y, direction)
    mono_centered = 2 * mono - 1

    if y_history is not None:
        br = breakout_ratio(y, y_history, direction)
        br_scaled = float(np.clip(br * 4.0, 0, 4))

        exc = exceedance_fraction(y, y_history, direction)
        exc_scaled = 2 * exc - 1

        return float(
            w_slope * norm_slope
            + w_endpoint * norm_ep
            + w_monotonicity * mono_centered
            + w_breakout * br_scaled
            + w_exceedance * exc_scaled
        )

    total = w_slope + w_endpoint + w_monotonicity
    return float(
        (w_slope / total) * norm_slope
        + (w_endpoint / total) * norm_ep
        + (w_monotonicity / total) * mono_centered
    )


def trend_shift(
    y_counterfactual: np.ndarray,
    y_original: np.ndarray,
    direction: str = "up",
) -> float:
    """How much did the counterfactual shift the slope vs. the original?

    Positive = shifted in the desired *direction*.
    """
    sign = 1.0 if direction == "up" else -1.0
    return float((linear_slope(y_counterfactual) - linear_slope(y_original)) * sign)


def percent_above_original(
    y_counterfactual: np.ndarray,
    y_original: np.ndarray,
    direction: str = "up",
) -> float:
    """Fraction of timesteps where the counterfactual exceeds (or undercuts)
    the original forecast, depending on *direction*.
    """
    if direction == "up":
        return float(np.mean(y_counterfactual > y_original))
    return float(np.mean(y_counterfactual < y_original))
