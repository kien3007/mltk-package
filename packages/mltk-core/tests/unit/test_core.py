"""Unit tests for MLTK Core functionality."""

import json

import pandas as pd
import pytest
from mltk_core.base import ColumnType, infer_column_types
from mltk_core.check import CheckResult, Report
from mltk_core.contract import ContractViolation, define_schema, validate_contract
from mltk_core.exceptions import SchemaError
from mltk_core.logging import get_logger, get_run_id, set_run_id
from mltk_core.metadata import ArtifactMetadata, compute_data_hash


def test_infer_column_types():
    df = pd.DataFrame(
        {
            "num_int": [1, 2, 3],
            "num_float": [1.1, 2.2, 3.3],
            "cat_str": ["a", "b", "c"],
            "cat_bool": [True, False, True],
            "dt": pd.date_range("2026-01-01", periods=3),
        }
    )
    types = infer_column_types(df)
    assert types["num_int"] == ColumnType.NUMERIC
    assert types["num_float"] == ColumnType.NUMERIC
    assert types["cat_str"] == ColumnType.CATEGORICAL
    assert types["cat_bool"] == ColumnType.CATEGORICAL
    assert types["dt"] == ColumnType.DATETIME


def test_report_logic():
    r1 = CheckResult(name="check1", level="INFO", passed=True, message="All good")
    r2 = CheckResult(name="check2", level="WARNING", passed=False, message="Slight drift")
    report = Report([r1, r2])
    assert report.passed is True
    assert report.has_warnings is True
    assert report.has_errors is False

    # Adding an error
    r3 = CheckResult(name="check3", level="ERROR", passed=False, message="Critical schema mismatch")
    report.add(r3)
    assert report.passed is False
    assert report.has_errors is True

    # Test raise_if_failed
    with pytest.raises(SchemaError):
        report.raise_if_failed()

    # Test HTML and JSON outputs
    html = report.to_html()
    assert "Critical schema mismatch" in html
    json_str = report.to_json()
    data = json.loads(json_str)
    assert data["errors"] == 1


def test_logging_and_run_id():
    set_run_id("test-run-123")
    assert get_run_id() == "test-run-123"

    logger = get_logger("test_logger")
    assert logger.name == "test_logger"


def test_metadata_and_hash():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    h1 = compute_data_hash(df)
    h2 = compute_data_hash(df.copy())
    assert h1 == h2

    meta = ArtifactMetadata(data_hash=h1, random_state=42)
    d = meta.to_dict()
    assert d["data_hash"] == h1
    assert d["random_state"] == 42


def test_data_contract():
    import pandera as pa

    schema = define_schema(
        columns={
            "age": pa.Int,
            "salary": pa.Float,
        }
    )
    valid_df = pd.DataFrame({"age": [25, 30], "salary": [50000.0, 75000.0]})
    out = validate_contract(valid_df, schema)
    assert len(out) == 2

    invalid_df = pd.DataFrame({"age": ["twenty-five", "thirty"], "salary": [50000.0, 75000.0]})
    with pytest.raises(ContractViolation):
        validate_contract(invalid_df, schema)
