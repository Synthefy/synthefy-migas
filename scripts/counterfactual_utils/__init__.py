"""Counterfactual scenario utilities for Migas-1.5.

Public API — import from the notebook with::

    import sys; sys.path.insert(0, "..")
    from scripts.counterfactual_utils import (
        run_baseline, run_best_of_n, results_to_dataframe,
        plot_scenario_comparison, plot_scenario_grid, plot_trend_summary,
        display_text_comparison,
        splice_summary, extract_factual, extract_predictive,
        composite_trend_score, linear_slope, trend_shift,
    )
"""

from .trend_metrics import (
    composite_trend_score,
    linear_slope,
    endpoint_change,
    monotonicity,
    breakout_ratio,
    exceedance_fraction,
    trend_shift,
    percent_above_original,
)
from .text_generation import (
    extract_factual,
    extract_predictive,
    splice_summary,
    build_scenario_prompt,
    generate_scenario_texts,
)
from .scenario_pipeline import (
    ScenarioResult,
    results_to_dataframe,
    run_baseline,
    run_best_of_n,
)
from .plotting import (
    plot_scenario_comparison,
    plot_scenario_grid,
    plot_trend_summary,
    display_text_comparison,
)

__all__ = [
    "composite_trend_score",
    "linear_slope",
    "endpoint_change",
    "monotonicity",
    "breakout_ratio",
    "exceedance_fraction",
    "trend_shift",
    "percent_above_original",
    "extract_factual",
    "extract_predictive",
    "splice_summary",
    "build_scenario_prompt",
    "generate_scenario_texts",
    "ScenarioResult",
    "results_to_dataframe",
    "run_baseline",
    "run_best_of_n",
    "plot_scenario_comparison",
    "plot_scenario_grid",
    "plot_trend_summary",
    "display_text_comparison",
]
