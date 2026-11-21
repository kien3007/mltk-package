"""Data contract helper using Pandera."""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    import pandera.pandas as pa
except ImportError:
    import pandera as pa

from mltk_core.exceptions import ContractViolation


def define_schema(
    columns: dict[str, Any] | None = None,
    checks: list[Any] | None = None,
    strict: bool = False,
    coerce: bool = False,
) -> pa.DataFrameSchema:
    """
    Define a Pandera DataFrameSchema with standard defaults for ML tabular data.
    """
    cols_dict = {}
    if columns:
        for col_name, dtype in columns.items():
            if isinstance(dtype, pa.Column):
                cols_dict[col_name] = dtype
            else:
                cols_dict[col_name] = pa.Column(dtype)

    return pa.DataFrameSchema(
        columns=cols_dict,
        checks=checks,
        strict=strict,
        coerce=coerce,
    )


def validate_contract(df: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """
    Validate a DataFrame against a Pandera schema, raising ContractViolation on failure.
    """
    try:
        return schema.validate(df)
    except pa.errors.SchemaError as exc:
        raise ContractViolation(f"Data contract violation: {exc}") from exc
