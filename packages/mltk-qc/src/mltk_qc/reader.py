"""Robust CSV reader with automatic encoding, delimiter, and type detection."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any
import chardet
import pandas as pd

from mltk_core.exceptions import SchemaError


def detect_encoding(filepath: str | Path, sample_size: int = 65536) -> str:
    """Detect file character encoding using chardet."""
    with open(filepath, "rb") as f:
        raw = f.read(sample_size)
    if not raw:
        return "utf-8"
    result = chardet.detect(raw)
    return result.get("encoding") or "utf-8"


def detect_delimiter(filepath: str | Path, encoding: str, sample_size: int = 8192) -> str:
    """Sniff the CSV delimiter from the first sample bytes."""
    with open(filepath, "r", encoding=encoding, errors="replace") as f:
        sample = f.read(sample_size)
    if not sample:
        return ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        # Fallback to frequency count heuristic
        counts = {d: sample.count(d) for d in [",", ";", "\t", "|"]}
        best = max(counts, key=counts.get)  # type: ignore[arg-type]
        return best if counts[best] > 0 else ","


def read_csv(
    filepath: str | Path,
    encoding: str | None = None,
    sep: str | None = None,
    clean_column_names: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Read a CSV file into a pandas DataFrame with robust automatic detection.

    Features:
    - Automatic character encoding detection (UTF-8, Latin-1, CP1252, etc.)
    - Automatic delimiter sniffing (comma, semicolon, tab, pipe)
    - Trims leading/trailing whitespace in column headers
    - Throws clear SchemaError if file is empty or corrupted
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.stat().st_size == 0:
        raise SchemaError(f"CSV file is completely empty: {path}")

    enc = encoding or detect_encoding(path)
    delimiter = sep or detect_delimiter(path, enc)

    try:
        df = pd.read_csv(path, encoding=enc, sep=delimiter, **kwargs)
    except Exception as exc:
        # If encoding failed, try utf-8-sig or latin-1 fallback
        try:
            df = pd.read_csv(path, encoding="latin-1", sep=delimiter, **kwargs)
        except Exception:
            raise SchemaError(f"Failed to read CSV at {path}: {exc}") from exc

    if clean_column_names:
        df.columns = [str(c).strip() for c in df.columns]

    if df.empty and len(df.columns) == 0:
        raise SchemaError(f"No valid data or columns parsed from CSV: {path}")

    return df
