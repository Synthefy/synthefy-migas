"""Baseline registry - central configuration for all evaluation baselines."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class BaselineConfig:
    """Configuration for a single baseline."""

    eval_func: Callable
    prediction_keys: List[str]
    help_text: str = ""
    requires_model: bool = False
    model_attr: Optional[str] = None
    extra_args_map: Dict[str, str] = field(default_factory=dict)
    depends_on: Optional[str] = None


BASELINE_REGISTRY: Dict[str, BaselineConfig] = {}


def register_baseline(
    name: str,
    eval_func: Callable,
    prediction_keys: List[str],
    help_text: str = "",
    requires_model: bool = False,
    model_attr: Optional[str] = None,
    extra_args_map: Optional[Dict[str, str]] = None,
    depends_on: Optional[str] = None,
):
    """Register a baseline for evaluation."""
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
    """Return all possible prediction keys from all baselines."""
    keys = []
    for config in BASELINE_REGISTRY.values():
        keys.extend(config.prediction_keys)
    return list(set(keys))


def get_baseline_for_prediction_key(key: str) -> Optional[str]:
    """Return the baseline name that produces a given prediction key."""
    for name, config in BASELINE_REGISTRY.items():
        if key in config.prediction_keys:
            return name
    return None
