"""Baseline registry - central configuration for all evaluation baselines."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class BaselineConfig:
    """Configuration for a single baseline.

    Attributes:
        eval_func: Callable that runs the baseline (loader, device, **kwargs) -> dict.
        prediction_keys: List of keys written in the returned "predictions" dict.
        help_text: Short description for CLI --help.
        requires_model: If True, a loaded model is passed to eval_func.
        model_attr: Key in the models dict to pass as first positional arg (e.g. "model").
        extra_args_map: Map from eval_func kwarg name to CLI args attribute name.
        depends_on: Name of another baseline that must run first (e.g. chronos2_gpt depends_on gpt_forecast).
    """

    eval_func: Callable
    prediction_keys: List[str]
    help_text: str = ""
    requires_model: bool = False
    model_attr: Optional[str] = None
    extra_args_map: Dict[str, str] = field(default_factory=dict)
    depends_on: Optional[str] = None


BASELINE_REGISTRY: Dict[str, BaselineConfig] = {}

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "ttfm": "TTFM",
    "ttfm_timesfm": "TTFM-TimesFM",
    "timeseries": "TS-Only",
    "chronos_univar": "Chronos2",
    "chronos_multivar": "Chronos2-MV",
    "chronos_emb": "Chronos2-MV",
    "chronos_gpt_cov": "Chronos2-GPT",
    "chronos_gpt_dir_cov": "Chronos2-GPT-MD",
    "chronos_naive_cov": "Chronos2-Naive",
    "timesfm_univar": "TimesFM2.5",
    "gpt_forecast": "GPT-OSS",
    "tabpfn_ts": "TabPFN2.5",
    "prophet": "Prophet",
    "naive": "Naive",
    "toto_univar": "Toto",
    "toto_emb": "Toto-MV",
    "migas": "Migas",
}


def get_display_name(key: str) -> str:
    """Human-readable label for a model prediction key.

    Falls back to replacing underscores with spaces if the key is unknown.
    """
    return MODEL_DISPLAY_NAMES.get(key, key.replace("_", " ").strip())


def register_baseline(
    name: str,
    eval_func: Callable,
    prediction_keys: List[str],
    help_text: str = "",
    requires_model: bool = False,
    model_attr: Optional[str] = None,
    extra_args_map: Optional[Dict[str, str]] = None,
    depends_on: Optional[str] = None,
) -> None:
    """Register a baseline for evaluation.

    Args:
        name: Unique baseline name (e.g. "chronos2", "gpt_forecast").
        eval_func: Function (loader, device, **kwargs) -> dict with input, gt, predictions.
        prediction_keys: Keys that eval_func writes in result["predictions"].
        help_text: Description shown for --eval_<name> in CLI. Defaults to "".
        requires_model: If True, a model is loaded and passed to eval_func. Defaults to False.
        model_attr: Key in models dict for the model (e.g. "model"). Defaults to None.
        extra_args_map: Map eval_func kwarg -> args attribute for CLI. Defaults to None.
        depends_on: Name of baseline that must be run first. Defaults to None.
    """
    BASELINE_REGISTRY[name] = BaselineConfig(
        eval_func=eval_func,
        prediction_keys=prediction_keys,
        help_text=help_text,
        requires_model=requires_model,
        model_attr=model_attr,
        extra_args_map=extra_args_map or {},
        depends_on=depends_on,
    )


def get_all_prediction_keys() -> List[str]:
    """Return all possible prediction keys from all registered baselines.

    Returns:
        Sorted list of unique prediction keys (e.g. ["chronos_univar", "ttfm", ...]).
    """
    keys = []
    for config in BASELINE_REGISTRY.values():
        keys.extend(config.prediction_keys)
    return list(set(keys))


def get_baseline_for_prediction_key(key: str) -> Optional[str]:
    """Return the baseline name that produces a given prediction key.

    Args:
        key: Prediction key (e.g. "ttfm", "chronos_univar").

    Returns:
        Baseline name that registers that key, or None if not found.
    """
    for name, config in BASELINE_REGISTRY.items():
        if key in config.prediction_keys:
            return name
    return None
