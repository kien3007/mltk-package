"""Contract tests ensuring every transformer complies with MLTK requirements."""

from __future__ import annotations

import pandas as pd
import pytest
from mltk_core.exceptions import NotFittedError


def test_fit_transform_returns_dataframe(transformer_entry, sample_df, sample_y):
    """Transformer must return a pandas DataFrame from fit_transform."""
    try:
        t = transformer_entry()
    except TypeError:
        pytest.skip(f"{transformer_entry.__name__} requires constructor arguments.")

    result = t.fit_transform(sample_df, sample_y)
    assert isinstance(result, pd.DataFrame), f"{transformer_entry} did not return a DataFrame."
    assert len(result) == len(sample_df), "Result row count must match input row count."


def test_does_not_mutate_input(transformer_entry, sample_df, sample_y):
    """Transformer must never mutate the input DataFrame in place."""
    try:
        t = transformer_entry()
    except TypeError:
        pytest.skip(f"{transformer_entry.__name__} requires constructor arguments.")

    original = sample_df.copy()
    _ = t.fit(sample_df, sample_y)
    pd.testing.assert_frame_equal(sample_df, original)

    _ = t.transform(sample_df)
    pd.testing.assert_frame_equal(sample_df, original)


def test_raises_not_fitted_error(transformer_entry, sample_df):
    """Transformer must raise NotFittedError if transform is called before fit."""
    try:
        t = transformer_entry()
    except TypeError:
        pytest.skip(f"{transformer_entry.__name__} requires constructor arguments.")

    with pytest.raises(NotFittedError):
        t.transform(sample_df)


def test_deterministic(transformer_entry, sample_df, sample_y):
    """Given random_state, two fit_transforms on the same data must produce identical outputs."""
    try:
        t1 = transformer_entry(random_state=42)
        t2 = transformer_entry(random_state=42)
    except TypeError:
        try:
            t1 = transformer_entry()
            t2 = transformer_entry()
        except TypeError:
            pytest.skip(f"{transformer_entry.__name__} constructor not standard.")

    r1 = t1.fit_transform(sample_df, sample_y)
    r2 = t2.fit_transform(sample_df, sample_y)
    pd.testing.assert_frame_equal(r1, r2)
