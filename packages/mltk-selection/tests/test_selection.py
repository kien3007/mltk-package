"""Unit tests for mltk-selection."""

import numpy as np
import pandas as pd
import pytest

from mltk_selection import (
    CorrelationFilter,
    MissingRateFilter,
    ModelImportanceSelector,
    SHAPSelector,
    VarianceFilter,
)


def test_variance_filter():
    df = pd.DataFrame({"const": [1.0, 1.0, 1.0, 1.0], "var": [1.0, 2.0, 3.0, 4.0]})
    vf = VarianceFilter(threshold=0.0)
    res = vf.fit_transform(df)
    assert "const" not in res.columns
    assert "var" in res.columns


def test_missing_rate_filter():
    df = pd.DataFrame({"mostly_null": [np.nan, np.nan, np.nan, 1.0], "clean": [1.0, 2.0, 3.0, 4.0]})
    mf = MissingRateFilter(threshold=0.5)
    res = mf.fit_transform(df)
    assert "mostly_null" not in res.columns
    assert "clean" in res.columns


def test_correlation_filter():
    x = np.linspace(0, 10, 50)
    df = pd.DataFrame({"feat_a": x, "feat_b": x * 2.0 + 0.0001, "feat_c": np.sin(x)})
    cf = CorrelationFilter(threshold=0.98)
    res = cf.fit_transform(df)
    assert "feat_a" in res.columns
    assert "feat_b" not in res.columns  # Collinear with feat_a
    assert "feat_c" in res.columns


def test_model_importance_selector():
    np.random.seed(42)
    n = 100
    x_signal = np.random.randn(n)
    x_noise = np.random.randn(n)
    y = (x_signal > 0).astype(int)

    df = pd.DataFrame({"signal": x_signal, "noise": x_noise})
    selector = ModelImportanceSelector(top_k=1, random_state=42)
    res = selector.fit_transform(df, y)

    assert "signal" in res.columns
    assert "noise" not in res.columns


def test_shap_selector():
    np.random.seed(42)
    n = 80
    x_signal = np.random.randn(n)
    x_noise = np.random.randn(n)
    y = (x_signal > 0).astype(int)

    df = pd.DataFrame({"signal": x_signal, "noise": x_noise})
    selector = SHAPSelector(top_k=1, random_state=42)
    res = selector.fit_transform(df, y)

    assert "signal" in res.columns
    assert "noise" not in res.columns
