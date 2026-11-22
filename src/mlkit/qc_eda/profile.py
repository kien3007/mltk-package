"""Dataset profiling and semantic type inference.

This is the shared "data understanding" layer: QC checks, EDA, charts and preprocessing all
read the same :class:`DatasetProfile` instead of re-guessing column types on their own, and
all type conversions go through one function, :func:`apply_coercion`.

Semantic types (``ColumnProfile.semantic_type``):

=================  ===============================================================
empty              no non-missing values
constant           a single distinct value
boolean            bool dtype, or two values such as yes/no, true/false, y/n
binary             numeric or categorical with exactly two distinct values
numeric            numbers, incl. numbers stored as text ("1,234", "$5", "12%") and
                   durations (timedelta -> seconds)
datetime           datetime dtype (any timezone), date text, or dates encoded as numbers
                   (yyyymmdd ints; epoch / Excel serials when the name says it is a date)
categorical        text/category with <= ``high_cardinality`` distinct values; also codes
                   with leading zeros ("00123") that must not become numbers
high_cardinality   text/category with more distinct values than that
identifier         (almost) unique per row: keys, codes, UUIDs, phone numbers
text               free text (long strings / many words)
ignored            excluded by the user (``column_types={"col": "ignore"}``)
unsupported        could not be analysed (the reason is in ``notes``); never crashes callers
=================  ===============================================================

Inference can be overridden per column with ``ProfileOptions(overrides={...})`` (or
``TaskConfig.column_types``); every consumer honours the same overrides. Accepted values:
``numeric``, ``categorical``, ``boolean``, ``text``, ``identifier``, ``ignore``, ``datetime``
and the explicit date encodings ``datetime:yyyymmdd``, ``datetime:epoch_s``,
``datetime:epoch_ms``, ``datetime:excel`` or ``datetime:<strftime format>``.
Columns whose type is a judgement call get an ``"ambiguous: ..."`` note.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

SEMANTIC_TYPES = (
    "empty",
    "constant",
    "boolean",
    "binary",
    "numeric",
    "datetime",
    "categorical",
    "high_cardinality",
    "identifier",
    "text",
    "ignored",
    "unsupported",
)
OVERRIDE_TYPES = ("numeric", "categorical", "boolean", "datetime", "text", "identifier", "ignore")
DATE_ENCODINGS = ("yyyymmdd", "epoch_s", "epoch_ms", "excel")

_BOOL_TOKENS = {"true", "false", "yes", "no", "y", "n", "t", "f", "0", "1", "on", "off"}
_ID_TOKENS = {"id", "key", "uuid", "guid", "pk"}
_TIME_TOKENS = {
    "date",
    "time",
    "timestamp",
    "ts",
    "day",
    "dt",
    "at",
    "created",
    "updated",
    "datetime",
}
_CURRENCY = re.compile(r"[$€£¥₫]")
_THOUSANDS = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")
_LEADING_ZERO = re.compile(r"^0\d")


def _name_tokens(name: str) -> list[str]:
    """Split ``customer_id`` / ``customerID`` / ``CustomerId`` into lowercase word tokens."""
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)  # "HTTPServer" -> "HTTP Server"
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", spaced)  # "customerID" -> "customer ID"
    return [t.lower() for t in re.split(r"[^A-Za-z0-9]+", spaced) if t]


def name_looks_like_id(name: str) -> bool:
    return any(t in _ID_TOKENS for t in _name_tokens(str(name)))


def name_looks_like_time(name: str) -> bool:
    return any(t in _TIME_TOKENS for t in _name_tokens(str(name)))


def validate_override(kind: str) -> str:
    base, _, fmt = str(kind).partition(":")
    if base not in OVERRIDE_TYPES or (fmt and base != "datetime"):
        raise ValueError(
            f"Unknown column type override {kind!r}. Use one of {list(OVERRIDE_TYPES)} or "
            f"'datetime:<{'|'.join(DATE_ENCODINGS)}|strftime format>'."
        )
    return str(kind)


@dataclass
class ProfileOptions:
    high_cardinality: int = 50
    identifier_unique_ratio: float = 0.95
    min_rows_for_identifier: int = 50
    text_min_avg_len: float = 30.0
    text_min_avg_words: float = 4.0
    parse_rate: float = 0.95
    datetime_parse_rate: float = 0.9
    leading_zero_share: float = 0.05
    sample_size: int = 2000
    random_state: int = 0
    overrides: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for kind in self.overrides.values():
            validate_override(kind)


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    semantic_type: str
    n_missing: int
    missing_rate: float
    n_unique: int
    unique_ratio: float
    dominant_share: float
    top_values: list[tuple[str, int]] = field(default_factory=list)
    #: how to convert the raw values (see :func:`apply_coercion`): "numeric", "boolean",
    #: "datetime", "timedelta", "category", one of DATE_ENCODINGS or "datetime:<fmt>"
    coercion: str | None = None
    numeric: dict[str, float] = field(default_factory=dict)
    datetime: dict[str, str] = field(default_factory=dict)
    text: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    overridden: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def ambiguous(self) -> bool:
        return any(n.startswith("ambiguous") for n in self.notes)


@dataclass
class TargetProfile:
    name: str
    kind: str  # "binary" | "multiclass" | "regression" | "constant" | "empty"
    n_missing: int
    class_counts: dict[str, int] = field(default_factory=dict)
    minority_share: float | None = None
    numeric: dict[str, float] = field(default_factory=dict)


@dataclass
class DatasetProfile:
    n_rows: int
    n_cols: int
    memory_mb: float
    n_duplicate_rows: int
    duplicate_columns: list[list[str]]
    columns: dict[str, ColumnProfile]
    target: TargetProfile | None = None
    duplicate_names: list[str] = field(default_factory=list)

    def columns_of(self, *types: str) -> list[str]:
        return [c for c, p in self.columns.items() if p.semantic_type in types]

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for p in self.columns.values():
            rows.append(
                {
                    "column": p.name,
                    "type": p.semantic_type,
                    "dtype": p.dtype,
                    "coercion": p.coercion,
                    "missing_rate": p.missing_rate,
                    "n_unique": p.n_unique,
                    "dominant_share": p.dominant_share,
                    "mean": p.numeric.get("mean"),
                    "std": p.numeric.get("std"),
                    "min": p.numeric.get("min"),
                    "median": p.numeric.get("p50"),
                    "max": p.numeric.get("max"),
                    "skew": p.numeric.get("skew"),
                    "outlier_rate": p.numeric.get("outlier_rate"),
                    "top": ", ".join(f"{v} ({c})" for v, c in p.top_values[:3]),
                    "notes": "; ".join(p.notes),
                }
            )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "memory_mb": self.memory_mb,
            "n_duplicate_rows": self.n_duplicate_rows,
            "duplicate_columns": self.duplicate_columns,
            "duplicate_names": self.duplicate_names,
            "columns": {c: p.to_dict() for c, p in self.columns.items()},
            "target": None if self.target is None else asdict(self.target),
        }


# --------------------------------------------------------------------------- coercion
#: Strings that mean "missing" in exported data (compared case-insensitively, stripped).
MISSING_TOKENS = frozenset(
    {"", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?", "missing", "#n/a", "#null!"}
)


def disguised_missing(values: pd.Series) -> pd.Series:
    """Boolean mask of text values that are really missing (``"n/a"``, ``"null"``, ``"-"``...)."""
    if (
        pd.api.types.is_numeric_dtype(values)
        or pd.api.types.is_datetime64_any_dtype(values)
        or pd.api.types.is_timedelta64_dtype(values)
    ):
        return pd.Series(False, index=values.index)
    try:
        s = values.astype("string").str.strip().str.lower()
    except (TypeError, ValueError):
        return pd.Series(False, index=values.index)
    return s.isin(MISSING_TOKENS).fillna(False).astype(bool) & values.notna()


def normalize_missing(values: pd.Series) -> pd.Series:
    """Replace disguised missing tokens with NaN (numeric/datetime columns are returned as-is)."""
    mask = disguised_missing(values)
    return values.mask(mask) if mask.any() else values


def coerce_numeric(values: pd.Series) -> pd.Series:
    """Parse numbers stored as text: thousands separators, currency symbols, %, decimal commas.

    Internal whitespace is *not* removed, so phone numbers such as "0912 345 678" stay text.
    """
    s = values.astype("string").str.strip()
    s = s.str.replace(_CURRENCY, "", regex=True).str.strip().str.rstrip("%")
    thousands = s.str.match(_THOUSANDS, na=False)
    s = s.where(~thousands, s.str.replace(",", "", regex=False))
    decimal_comma = s.str.match(r"^[+-]?\d+,\d+$", na=False)
    s = s.where(~decimal_comma, s.str.replace(",", ".", regex=False))
    # plain float64 with NaN (not the nullable Float64/pd.NA that string dtype would give)
    return pd.to_numeric(s, errors="coerce").astype("float64")


def coerce_boolean(values: pd.Series) -> pd.Series:
    """Map yes/no-like tokens to 1.0/0.0 (NaN for anything else)."""
    if pd.api.types.is_bool_dtype(values):
        return values.astype("float64")
    s = values.astype("string").str.strip().str.lower()
    mapping = {"true": 1.0, "yes": 1.0, "y": 1.0, "t": 1.0, "1": 1.0, "on": 1.0}
    mapping.update({"false": 0.0, "no": 0.0, "n": 0.0, "f": 0.0, "0": 0.0, "off": 0.0})
    return s.map(mapping).astype(float)


def _naive_utc(dt: pd.Series) -> pd.Series:
    """Timezone-aware -> naive UTC, so every datetime column has one comparable dtype."""
    if isinstance(dt.dtype, pd.DatetimeTZDtype):
        return dt.dt.tz_convert("UTC").dt.tz_localize(None)
    return dt


def coerce_datetime(values: pd.Series, encoding: str | None = None) -> pd.Series:
    """Parse dates to naive (UTC) ``datetime64``.

    ``encoding``: None/"datetime" (auto-parse text, any timezone), "yyyymmdd", "epoch_s",
    "epoch_ms", "excel" (days since 1899-12-30) or "datetime:<strftime format>".
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        return _naive_utc(values)
    enc = encoding or "datetime"
    if enc == "yyyymmdd":
        digits = pd.to_numeric(values, errors="coerce").round().astype("Int64").astype("string")
        return pd.to_datetime(digits, format="%Y%m%d", errors="coerce")
    if enc in ("epoch_s", "epoch_ms"):
        num = pd.to_numeric(values, errors="coerce")
        return pd.to_datetime(num, unit="s" if enc == "epoch_s" else "ms", errors="coerce")
    if enc == "excel":
        num = pd.to_numeric(values, errors="coerce")
        return pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")
    text = values.astype("string")
    if enc.startswith("datetime:"):
        return _naive_utc(
            pd.to_datetime(text, format=enc.split(":", 1)[1], errors="coerce", utc=True)
        )
    with np.errstate(all="ignore"):
        try:
            out = pd.to_datetime(text, errors="coerce", format="mixed")
        except (ValueError, TypeError):  # e.g. mixed timezone offsets
            out = pd.to_datetime(text, errors="coerce", format="mixed", utc=True)
    return _naive_utc(out)


def apply_coercion(values: pd.Series, semantic_type: str, coercion: str | None) -> pd.Series:
    """Convert raw values according to a column profile (single source of truth).

    Returns float64 for numeric/boolean/duration columns, naive datetime64 for dates, and the
    original values (missing tokens normalised) for everything else.
    """
    s = normalize_missing(values)
    if (
        semantic_type == "datetime"
        or (coercion or "").startswith("datetime:")
        or coercion in DATE_ENCODINGS
    ):
        return coerce_datetime(s, coercion)
    if coercion == "timedelta" or pd.api.types.is_timedelta64_dtype(s):
        return pd.to_timedelta(s, errors="coerce").dt.total_seconds().astype("float64")
    if coercion == "boolean" or (
        semantic_type == "boolean" and not pd.api.types.is_numeric_dtype(s)
    ):
        return coerce_boolean(s)
    if coercion == "numeric":
        return coerce_numeric(s).replace([np.inf, -np.inf], np.nan)
    if coercion == "category":
        return s.where(s.isna(), s.astype(str)).astype(object)
    if pd.api.types.is_numeric_dtype(s) and semantic_type in ("numeric", "binary", "boolean"):
        return (
            pd.to_numeric(s, errors="coerce").astype("float64").replace([np.inf, -np.inf], np.nan)
        )
    return s


# --------------------------------------------------------------------------- inference
def _sample(s: pd.Series, opts: ProfileOptions) -> pd.Series:
    nn = s.dropna()
    if len(nn) > opts.sample_size:
        nn = nn.sample(opts.sample_size, random_state=opts.random_state)
    return nn


def _numeric_stats(x: pd.Series) -> dict[str, float]:
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    n_inf = int(np.isinf(arr).sum())
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"n_inf": float(n_inf)}
    q = np.percentile(arr, [1, 25, 50, 75, 99])
    iqr = q[3] - q[1]
    lo, hi = q[1] - 3 * iqr, q[3] + 3 * iqr
    std = float(arr.std())
    return {
        "mean": float(arr.mean()),
        "std": std,
        "min": float(arr.min()),
        "p01": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p99": float(q[4]),
        "max": float(arr.max()),
        "skew": float(pd.Series(arr).skew()) if len(arr) > 2 and std > 0 else 0.0,
        "zero_rate": float(np.mean(arr == 0)),
        "negative_rate": float(np.mean(arr < 0)),
        "outlier_rate": float(np.mean((arr < lo) | (arr > hi))) if iqr > 0 else 0.0,
        "is_integer": float(np.all(np.mod(arr, 1) == 0)),
        "n_inf": float(n_inf),
    }


def _date_encoding(nn: pd.Series, name: str, opts: ProfileOptions) -> str | None:
    """Detect dates stored as numbers. yyyymmdd is distinctive enough to detect from values;
    epoch and Excel serials are only accepted when the column *name* says it is a date."""
    arr = pd.to_numeric(nn, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    lo, hi = float(arr.min()), float(arr.max())
    integer = bool(np.all(np.mod(arr, 1) == 0))
    if integer and lo >= 19000101 and hi <= 21001231:
        sample = pd.Series(arr[: opts.sample_size])
        if coerce_datetime(sample, "yyyymmdd").notna().mean() >= opts.parse_rate:
            return "yyyymmdd"
    if not name_looks_like_time(name):
        return None
    if lo >= 1e9 and hi <= 2.2e9:
        return "epoch_s"
    if lo >= 1e12 and hi <= 2.2e12:
        return "epoch_ms"
    if lo >= 20000 and hi <= 80000:
        return "excel"
    return None


def _apply_override(prof: ColumnProfile, s: pd.Series, nn: pd.Series, kind: str) -> ColumnProfile:
    base, _, fmt = kind.partition(":")
    prof.overridden = True
    prof.notes.append(f"type set by user: {kind}")
    if base == "ignore":
        prof.semantic_type = "ignored"
    elif base == "numeric":
        prof.semantic_type = "numeric"
        if not pd.api.types.is_numeric_dtype(s):
            prof.coercion = "timedelta" if pd.api.types.is_timedelta64_dtype(s) else "numeric"
        prof.numeric = _numeric_stats(apply_coercion(nn, "numeric", prof.coercion))
    elif base == "datetime":
        prof.semantic_type = "datetime"
        if fmt in DATE_ENCODINGS:
            prof.coercion = fmt
        elif fmt:
            prof.coercion = f"datetime:{fmt}"
        else:
            prof.coercion = "datetime"
        dt = coerce_datetime(nn, prof.coercion)
        prof.datetime = {"min": str(dt.min()), "max": str(dt.max())}
        bad = float(dt.isna().mean()) if len(dt) else 0.0
        if bad > 0.05:
            prof.notes.append(f"{bad:.0%} of values do not parse as {kind}")
    elif base == "boolean":
        prof.semantic_type = "boolean"
        prof.coercion = "boolean"
    elif base == "categorical":
        prof.semantic_type = "categorical" if prof.n_unique <= 50 else "high_cardinality"
        prof.coercion = "category"
    else:  # text, identifier
        prof.semantic_type = base
    return prof


def infer_column(
    s: pd.Series, name: str | None = None, opts: ProfileOptions | None = None
) -> ColumnProfile:
    """Profile one column and infer its semantic type. Never raises: unexpected data yields
    ``semantic_type="unsupported"`` with the reason in ``notes``."""
    opts = opts or ProfileOptions()
    name = str(s.name if name is None else name)
    try:
        return _infer(s, name, opts)
    except Exception as exc:  # noqa: BLE001 - robustness contract
        n = len(s)
        n_missing = int(pd.isna(s).sum())
        return ColumnProfile(
            name=name,
            dtype=str(getattr(s, "dtype", "?")),
            semantic_type="unsupported",
            n_missing=n_missing,
            missing_rate=n_missing / n if n else 0.0,
            n_unique=0,
            unique_ratio=0.0,
            dominant_share=0.0,
            notes=[f"could not be analysed: {type(exc).__name__}: {str(exc)[:120]}"],
        )


def _infer(s: pd.Series, name: str, opts: ProfileOptions) -> ColumnProfile:
    n = len(s)
    disguised = int(disguised_missing(s).sum())
    if disguised:
        s = normalize_missing(s)
    n_missing = int(s.isna().sum())
    nn = s.dropna()
    m = len(nn)
    try:
        counts = nn.value_counts()
    except TypeError:  # unhashable values (lists, dicts)
        counts = nn.astype(str).value_counts()
    n_unique = int(len(counts))
    prof = ColumnProfile(
        name=name,
        dtype=str(s.dtype),
        semantic_type="empty",
        n_missing=n_missing,
        missing_rate=n_missing / n if n else 0.0,
        n_unique=n_unique,
        unique_ratio=n_unique / m if m else 0.0,
        dominant_share=float(counts.iloc[0] / m) if m else 0.0,
        top_values=[(str(k), int(v)) for k, v in counts.head(10).items()],
    )
    if disguised:
        prof.notes.append(f"{disguised} disguised missing values (e.g. 'n/a', 'null', '-')")
    if name in opts.overrides:
        return _apply_override(prof, s, nn, opts.overrides[name])
    if m == 0:
        return prof
    if n_unique == 1:
        prof.semantic_type = "constant"
        return prof

    name_is_id = name_looks_like_id(name)
    if pd.api.types.is_bool_dtype(s):
        prof.semantic_type = "boolean"
        return prof
    if pd.api.types.is_datetime64_any_dtype(s):
        dt = _naive_utc(nn)
        prof.semantic_type = "datetime"
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            prof.coercion = "datetime"
            prof.notes.append(f"timezone {s.dtype.tz} converted to UTC")
        prof.datetime = {"min": str(dt.min()), "max": str(dt.max())}
        return prof
    if pd.api.types.is_timedelta64_dtype(s):
        prof.semantic_type = "numeric"
        prof.coercion = "timedelta"
        prof.numeric = _numeric_stats(pd.to_timedelta(nn).dt.total_seconds())
        prof.notes.append("duration converted to seconds")
        return prof
    if pd.api.types.is_numeric_dtype(s):
        prof.numeric = _numeric_stats(nn)
        is_int = bool(prof.numeric.get("is_integer", 0.0))
        encoding = _date_encoding(nn, name, opts) if n_unique > 2 else None
        if encoding:
            prof.semantic_type = "datetime"
            prof.coercion = encoding
            dt = coerce_datetime(nn, encoding)
            prof.datetime = {"min": str(dt.min()), "max": str(dt.max())}
            prof.notes.append(f"dates stored as numbers ({encoding})")
            return prof
        if n_unique == 2:
            prof.semantic_type = "binary"
        elif is_int and name_is_id and prof.unique_ratio >= 0.5:
            prof.semantic_type = "identifier"
        elif (
            is_int
            and m >= opts.min_rows_for_identifier
            and n_unique == m
            and nn.is_monotonic_increasing
        ):
            prof.semantic_type = "identifier"  # 1, 2, 3, ... row counters
        else:
            prof.semantic_type = "numeric"
            if is_int and 3 <= n_unique <= 10 and m >= 50:
                prof.notes.append(
                    "ambiguous: integer with few values - a count, or a category code? "
                    "set column_types to 'categorical' if it is a code"
                )
            elif name_looks_like_time(name) and is_int:
                prof.notes.append(
                    "ambiguous: name suggests a date but values are plain integers - "
                    "set column_types to 'datetime:<encoding>' if it is one"
                )
        if prof.numeric.get("n_inf", 0) > 0:
            prof.notes.append(f"{int(prof.numeric['n_inf'])} infinite values")
        return prof

    # ---- text-like / categorical (object, string, category, mixed)
    sample = _sample(s, opts)
    try:
        as_str = sample.astype(str)
    except (TypeError, ValueError):
        as_str = sample.map(str)
    leading_zero = float(as_str.str.strip().str.match(_LEADING_ZERO).mean()) if len(as_str) else 0.0
    num = coerce_numeric(sample)
    num_rate = float(num.notna().mean()) if len(sample) else 0.0
    if num_rate >= opts.parse_rate and leading_zero < opts.leading_zero_share:
        prof.coercion = "numeric"
        full = coerce_numeric(nn)
        prof.numeric = _numeric_stats(full.dropna())
        prof.semantic_type = "binary" if full.nunique() == 2 else "numeric"
        prof.notes.append("numbers stored as text")
        return prof
    if num_rate >= opts.parse_rate:
        prof.notes.append(
            f"codes with leading zeros ({leading_zero:.0%} of values, e.g. '00123') kept as text"
        )
    elif 0.3 <= num_rate < opts.parse_rate:
        prof.notes.append(f"mixed types: {num_rate:.0%} of values look numeric")

    lowered = set(as_str.str.strip().str.lower().unique())
    if n_unique == 2 and lowered <= _BOOL_TOKENS:
        prof.semantic_type = "boolean"
        prof.coercion = "boolean"
        return prof

    avg_len = float(as_str.str.len().mean())
    avg_words = float(as_str.str.count(r"\S+").mean())
    if num_rate < 0.5 and avg_len <= 40:
        dt = coerce_datetime(sample)
        if dt.notna().mean() >= opts.datetime_parse_rate:
            prof.semantic_type = "datetime"
            prof.coercion = "datetime"
            full = coerce_datetime(nn)
            prof.datetime = {"min": str(full.min()), "max": str(full.max())}
            return prof

    prof.text = {"avg_len": avg_len, "avg_words": avg_words}
    if name_is_id and prof.unique_ratio >= 0.9:
        prof.semantic_type = "identifier"  # named like a key and (almost) unique, even on tiny data
        return prof
    if (
        avg_len >= opts.text_min_avg_len or avg_words >= opts.text_min_avg_words
    ) and prof.unique_ratio > 0.5:
        prof.semantic_type = "text"
    elif m >= opts.min_rows_for_identifier and prof.unique_ratio >= opts.identifier_unique_ratio:
        prof.semantic_type = "identifier"
    elif n_unique == 2:
        prof.semantic_type = "binary"
    elif n_unique <= opts.high_cardinality:
        prof.semantic_type = "categorical"
    else:
        prof.semantic_type = "high_cardinality"
    return prof


def _profile_target(y: pd.Series, task_type: str | None) -> TargetProfile:
    nn = y.dropna()
    tp = TargetProfile(name=str(y.name), kind="empty", n_missing=int(y.isna().sum()))
    if nn.empty:
        return tp
    n_unique = nn.astype(str).nunique()
    if n_unique <= 1:
        tp.kind = "constant"
        return tp
    numeric = pd.api.types.is_numeric_dtype(nn) and not pd.api.types.is_bool_dtype(nn)
    regression = task_type == "regression" or (task_type is None and numeric and n_unique > 20)
    if regression:
        tp.kind = "regression"
        tp.numeric = _numeric_stats(pd.to_numeric(nn, errors="coerce").dropna())
        return tp
    counts = nn.astype(str).value_counts()
    tp.kind = "binary" if n_unique == 2 else "multiclass"
    tp.class_counts = {str(k): int(v) for k, v in counts.items()}
    tp.minority_share = float(counts.iloc[-1] / counts.sum())
    return tp


def _duplicate_columns(df: pd.DataFrame) -> list[list[str]]:
    hashes: dict[int, list[str]] = {}
    for c in df.columns:
        try:
            h = int(pd.util.hash_pandas_object(df[c], index=False).sum())
        except (TypeError, ValueError):
            continue
        hashes.setdefault(h, []).append(str(c))
    by_name = {str(c): c for c in df.columns}
    groups = []
    for cols in hashes.values():
        if len(cols) > 1:
            base = df[by_name[cols[0]]]
            same = [cols[0]] + [c for c in cols[1:] if df[by_name[c]].equals(base)]
            if len(same) > 1:
                groups.append(same)
    return groups


def deduplicate_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Keep the first of several columns sharing a name (also ``1`` vs ``"1"``).

    Returns the de-duplicated frame and the names that were duplicated.
    """
    names = pd.Index([str(c) for c in df.columns])
    dup = names.duplicated()
    if not dup.any():
        return df, []
    return df.loc[:, ~dup], sorted(set(names[dup]))


def profile_dataframe(
    df: pd.DataFrame,
    y: Any = None,
    task_type: str | None = None,
    options: ProfileOptions | None = None,
) -> DatasetProfile:
    """Profile every column of ``df`` (and the target ``y`` if given). Never raises on odd data:
    duplicate column names keep their first occurrence, unanalysable columns are 'unsupported'."""
    opts = options or ProfileOptions()
    df, dup_names = deduplicate_columns(df)
    try:
        n_dup = int(df.duplicated().sum()) if len(df.columns) else 0
    except (TypeError, ValueError):
        n_dup = int(df.astype(str).duplicated().sum())
    try:
        memory = float(df.memory_usage(deep=True).sum() / 1e6)
    except (TypeError, ValueError):
        memory = float("nan")
    by_name = {str(c): c for c in df.columns}
    prof = DatasetProfile(
        n_rows=len(df),
        n_cols=len(df.columns),
        memory_mb=memory,
        n_duplicate_rows=n_dup,
        duplicate_columns=_duplicate_columns(df) if len(df.columns) else [],
        columns={name: infer_column(df[col], name, opts) for name, col in by_name.items()},
        duplicate_names=dup_names,
    )
    if y is not None:
        y_s = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y), name="target")
        prof.target = _profile_target(y_s.reset_index(drop=True), task_type)
    return prof
