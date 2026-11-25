"""Modelling presets: sensible bundles of candidates / tuning / ensembling per time budget.

``modelling.preset`` fills only the keys you did not set yourself, e.g.::

    modelling:
      preset: best_quality
      time_budget: 600        # seconds for the whole model search (best effort)
      candidates: [linear, lightgbm]   # explicit keys always win over the preset
"""

from __future__ import annotations

import importlib.util
from typing import Any

from mlkit.core.errors import ValidationError

PRESETS: dict[str, dict[str, Any]] = {
    # quick baseline: two strong, cheap families, 3-fold CV, nothing extra
    "fast": {"candidates": ["linear", "lightgbm"], "cv_folds": 3},
    # default-like zoo plus a soft-voting ensemble of the best three
    "balanced": {
        "candidates": ["linear", "random_forest", "lightgbm"],
        "ensemble": "vote",
    },
    # widest zoo, Optuna tuning of the best family, stacking and calibration
    "best_quality": {
        "candidates": ["linear", "lightgbm", "hist_gb", "extra_trees", "random_forest"],
        "tuning": {"n_trials": 30},
        "ensemble": "stack",
        "calibration": "auto",
    },
}


def resolve_modelling(modelling: dict[str, Any] | None) -> dict[str, Any]:
    """Merge ``modelling.preset`` defaults under the user's explicit ``modelling`` keys."""
    user = dict(modelling or {})
    name = user.pop("preset", None)
    if name is None:
        return user
    if name not in PRESETS:
        raise ValidationError(f"Unknown modelling.preset {name!r}; choose one of {sorted(PRESETS)}")
    merged = {
        k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
        for k, v in PRESETS[name].items()
    }
    if name == "best_quality" and importlib.util.find_spec("xgboost") is not None:
        merged["candidates"] = [*merged["candidates"], "xgboost"]
    merged.update(user)
    merged["preset"] = name
    return merged
