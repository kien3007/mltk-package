"""Model Card generator for documentation and transparency."""

from __future__ import annotations

import os
from typing import Any
import yaml

from mltk_core.metadata import ArtifactMetadata


def generate_model_card(
    model_name: str,
    task_type: str,
    metrics: dict[str, Any],
    features: list[str],
    metadata: ArtifactMetadata,
    output_filepath: str | None = None,
) -> dict[str, Any]:
    """Generate structured model documentation (Model Card)."""
    card = {
        "model_details": {
            "name": model_name,
            "task_type": task_type,
            "created_at": metadata.created_at,
            "git_sha": metadata.git_sha,
            "run_id": metadata.run_id,
            "framework": "MLTK / scikit-learn",
        },
        "intended_use": {
            "domain": "Tabular batch prediction",
            "limitations": "Trained offline. Requires input features to match input schema.",
        },
        "features": {
            "feature_count": len(features),
            "feature_names": features,
        },
        "performance_metrics": metrics,
        "reproducibility": {
            "random_state": metadata.random_state,
            "data_hash": metadata.data_hash,
            "package_versions": metadata.package_versions,
        },
    }

    if output_filepath:
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        with open(output_filepath, "w", encoding="utf-8") as f:
            yaml.safe_dump(card, f, sort_keys=False)

    return card
