"""Pytest fixtures and dynamic transformer registration for contract testing."""

from __future__ import annotations

import importlib.metadata
from typing import Any

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Fixture providing a diverse sample DataFrame for contract testing."""
    return pd.DataFrame(
        {
            "num1": [1.0, 2.5, np.nan, 4.0, 100.0, 3.2, 5.0, 8.1, 9.0, 2.0],
            "num2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "cat1": ["a", "b", "a", "b", "c", "a", "c", "b", "a", "b"],
            "cat2": ["x", "y", "x", "z", "y", "x", "x", "z", "y", "x"],
        }
    )


@pytest.fixture
def sample_y() -> pd.Series:
    """Fixture providing binary target."""
    return pd.Series([0, 1, 0, 1, 0, 0, 1, 1, 0, 1], name="target")


def get_registered_transformers() -> list[tuple[str, Any]]:
    """Discover all transformers registered across MLTK packages via entry points."""
    transformers = []
    try:
        eps = importlib.metadata.entry_points()
        group_eps = (
            eps.select(group="mltk.transformers")
            if hasattr(eps, "select")
            else eps.get("mltk.transformers", [])
        )
        for ep in group_eps:
            try:
                cls = ep.load()
                transformers.append((ep.name, cls))
            except Exception:
                pass
    except Exception:
        pass
    return transformers


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize tests using the discovered transformer classes."""
    if "transformer_entry" in metafunc.fixturenames:
        entries = get_registered_transformers()
        if entries:
            metafunc.parametrize(
                "transformer_entry", [e[1] for e in entries], ids=[e[0] for e in entries]
            )
        else:
            # Fallback when entry-points haven't been installed in pip/uv yet
            metafunc.parametrize("transformer_entry", [])
