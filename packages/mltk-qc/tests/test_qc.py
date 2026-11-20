"""Unit tests for mltk-qc."""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from mltk_core.exceptions import SchemaError
from mltk_qc import read_csv, run_qc


def test_read_csv_auto_detection():
    # Test semicolon delimiter and encoding handling
    content = "col_a;col_b;col_c\n1;hello;10.5\n2;world;20.5\n"
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", encoding="utf-8") as f:
        f.write(content)
        path = f.name

    try:
        df = read_csv(path)
        assert list(df.columns) == ["col_a", "col_b", "col_c"]
        assert len(df) == 2
        assert df["col_b"].iloc[1] == "world"
    finally:
        Path(path).unlink(missing_ok=True)


def test_read_csv_empty_file():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        path = f.name

    try:
        with pytest.raises(SchemaError):
            read_csv(path)
    finally:
        Path(path).unlink(missing_ok=True)


def test_run_qc_checks():
    df = pd.DataFrame(
        {
            "const_col": [1, 1, 1, 1, 1],
            "missing_col": [None, None, None, None, 10.0],
            "card_col": ["a", "b", "c", "d", "e"],
            "leak_col": [0.0, 1.0, 0.0, 1.0, 0.0],
            "normal_col": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    target = pd.Series([0, 1, 0, 1, 0], name="target")

    report = run_qc(df, target=target)
    assert report.has_warnings is True
    assert report.has_errors is True  # Due to leak_col having 1.0 correlation with target!

    # Check leakage check caught leak_col
    leak_check = next(r for r in report.results if r.name == "data_leakage")
    assert leak_check.passed is False
    assert "leak_col" in leak_check.details["leaked_columns"]

    # Verify report HTML generation
    html = report.to_html()
    assert "data_leakage" in html
