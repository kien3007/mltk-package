"""MLTK-QC: Automated Quality Control and Exploratory Data Analysis."""

from __future__ import annotations

from mltk_qc.checks import (
    ConstantColumnCheck,
    DuplicateCheck,
    HighCardinalityCheck,
    LeakageCheck,
    MissingValueCheck,
    TargetDistributionCheck,
)
from mltk_qc.reader import detect_delimiter, detect_encoding, read_csv
from mltk_qc.runner import QCReport, run_qc

__all__ = [
    "ConstantColumnCheck",
    "DuplicateCheck",
    "HighCardinalityCheck",
    "LeakageCheck",
    "MissingValueCheck",
    "QCReport",
    "TargetDistributionCheck",
    "detect_delimiter",
    "detect_encoding",
    "read_csv",
    "run_qc",
]
