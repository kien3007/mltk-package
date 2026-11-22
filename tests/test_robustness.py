"""Robustness contract for the data foundation (profile, QC, EDA, charts, preprocessing).

1. Regression tests for edge cases that used to crash or be misread.
2. Shared semantic overrides (``column_types``) honoured everywhere.
3. Property-based fuzzing (hypothesis): for *any* generated table, profiling / QC / EDA never
   raise, and the preprocessing chain always yields model-ready data (numeric, finite, no NaN)
   whose columns are stable on a new batch drawn from the same schema.
"""

from __future__ import annotations

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlkit.core import TaskConfig
from mlkit.qc_eda import infer_column, profile_dataframe, run_eda, run_qc
from mlkit.qc_eda.profile import SEMANTIC_TYPES, ProfileOptions, apply_coercion

warnings.filterwarnings("ignore")


# =========================================================================== 1. edge cases
def _base(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.normal(size=n), "target": rng.integers(0, 2, n)})


def _edge_cases() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(1)
    n = 200
    b = _base(n)
    return {
        "nullable_dtypes": b.assign(
            i=pd.array(rng.integers(0, 50, n), dtype="Int64"),
            bo=pd.array(rng.choice([True, False, None], n), dtype="boolean"),
            s=pd.array(rng.choice(["a", "b", None], n), dtype="string"),
        ),
        "pd_na_object": b.assign(o=[pd.NA if i % 5 == 0 else f"v{i % 7}" for i in range(n)]),
        "tz_aware": b.assign(
            ts=pd.date_range("2024-01-01", periods=n, freq="h", tz="Asia/Ho_Chi_Minh")
        ),
        "mixed_tz_strings": b.assign(
            ts=["2024-01-01T10:00:00+07:00", "2024-01-02T10:00:00Z"] * (n // 2)
        ),
        "timedelta": b.assign(td=pd.to_timedelta(rng.integers(0, 1000, n), unit="m")),
        "duplicate_names": pd.concat([b, pd.DataFrame({"x": rng.normal(size=n)})], axis=1),
        "int_and_str_same_name": pd.concat(
            [b, pd.DataFrame({1: rng.normal(size=n), "1": rng.normal(size=n)})], axis=1
        ),
        "unicode_names": b.rename(columns={"x": " thu nhập (VNĐ) "}).assign(
            **{"Thành phố": rng.choice(["Hà Nội", "TP.HCM", "Đà Nẵng"], n)}
        ),
        "unused_categories": b.assign(
            c=pd.Categorical(rng.choice(["a", "b"], n), categories=list("abcd"))
        ),
        "list_dict_values": b.assign(
            l=[[i, i + 1] for i in range(n)], d=[{"k": i % 3} for i in range(n)]
        ),
        "huge_ints_narrow_range": b.assign(
            big=np.full(n, 2**62, dtype=np.int64) + rng.integers(0, 100, n)
        ),
        "uint64": b.assign(u=rng.integers(0, 2**32, n).astype(np.uint64)),
        "target_with_nan": b.assign(
            target=b["target"].astype(float).where(rng.uniform(size=n) > 0.3)
        ),
        "single_row": _base(1),
        "two_rows": _base(2),
        "no_features": b[["target"]],
        "bool_target": b.assign(target=b["target"].astype(bool)),
        "extreme_floats": b.assign(e=np.where(rng.uniform(size=n) < 0.5, 1e300, -0.0)),
        "leading_zero_codes": b.assign(zip=[f"{v:05d}" for v in rng.integers(0, 3000, n)]),
        "very_long_text": b.assign(
            t=[" ".join(["lorem"] * int(k)) for k in rng.integers(50, 300, n)]
        ),
        "emoji_categories": b.assign(c=rng.choice(["😀", "🔥", "ok"], n)),
        "all_nan_and_constant": b.assign(nan=np.nan, const="same"),
        "mixed_object_types": b.assign(
            m=[i if i % 3 == 0 else (f"s{i}" if i % 3 == 1 else None) for i in range(n)]
        ),
    }


@pytest.mark.parametrize("name", list(_edge_cases()))
def test_edge_case_never_crashes(name):
    from mlkit.preprocessing import TypeCaster

    df = _edge_cases()[name]
    report = run_eda(df, target="target" if "target" in df.columns else None)
    for _, fig in report.figures():
        plt.close(fig)
    assert "<html" in report.to_html()
    X = df.drop(columns=["target"], errors="ignore")
    out = TypeCaster().fit(X).transform(X)
    assert isinstance(out, pd.DataFrame)
    for p in report.profile.columns.values():
        assert p.semantic_type in SEMANTIC_TYPES


def test_duplicate_names_reported_and_first_kept():
    df = _edge_cases()["duplicate_names"]
    qc = run_qc(df.drop(columns="target"), y=df["target"], enforce_gate=False)
    assert any(i.name == "duplicate_names" for i in qc.report_.items)
    assert qc.profile_.duplicate_names == ["x"]


# ======================================================================= semantic meaning
@pytest.mark.parametrize(
    "name, values, expected_type, expected_coercion",
    [
        ("zip", [f"{v:05d}" for v in range(0, 600, 10)] * 5, "high_cardinality", None),
        (
            "phone",
            [f"09{v % 99:02d} {100 + v % 900} {200 + v % 700}" for v in range(300)],
            "identifier",
            None,
        ),
        ("duration", pd.to_timedelta(np.arange(300), unit="m"), "numeric", "timedelta"),
        (
            "order_day",
            [20240101, 20240215, 20231231, 20230705, 20220101] * 60,
            "datetime",
            "yyyymmdd",
        ),
        ("created_at", np.arange(1_600_000_000, 1_600_000_300), "datetime", "epoch_s"),
        ("invoice_date", np.arange(44000, 44300), "datetime", "excel"),
        (
            "amount",
            np.tile(np.arange(44000, 44150), 2)[::-1],
            "numeric",
            None,
        ),  # same values, no date hint
        (
            "ts",
            pd.date_range("2024-01-01", periods=300, freq="h", tz="UTC"),
            "datetime",
            "datetime",
        ),
        ("price", [f"${v:,.2f}" for v in np.linspace(10, 5000, 300)], "numeric", "numeric"),
    ],
)
def test_semantic_meaning(name, values, expected_type, expected_coercion):
    p = infer_column(pd.Series(values), name)
    assert (p.semantic_type, p.coercion) == (expected_type, expected_coercion), p.notes
    converted = apply_coercion(pd.Series(values), p.semantic_type, p.coercion)
    if expected_type == "datetime":
        assert pd.api.types.is_datetime64_dtype(converted) and converted.notna().all()
        assert converted.dt.tz is None  # always naive UTC


def test_ambiguous_columns_are_flagged_in_qc():
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"stars": rng.integers(1, 6, 300), "x": rng.normal(size=300)})
    assert infer_column(X["stars"], "stars").ambiguous
    qc = run_qc(X, y=pd.Series(rng.integers(0, 2, 300)), enforce_gate=False)
    assert "ambiguous_types" in {i.name for i in qc.report_.items if i.status == "warn"}


# ============================================================================ overrides
def _override_frame(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    stars = rng.integers(1, 6, n)
    return pd.DataFrame(
        {
            "stars": stars,
            "amount": rng.integers(44000, 45000, n),
            "notes": rng.choice(["a", "b"], n),
            "x": rng.normal(size=n),
            # non-monotone in stars (needs categorical treatment) + 10% label noise (not a leak)
            "target": np.where(rng.uniform(size=n) < 0.1, stars != 3, stars == 3).astype(int),
        }
    )


def test_overrides_validation():
    with pytest.raises(ValueError, match="Unknown column type"):
        ProfileOptions(overrides={"a": "number"})
    with pytest.raises(ValueError):
        TaskConfig(target="y", column_types={"a": "categorical:x"})
    assert TaskConfig(target="y", column_types={"a": "datetime:%d/%m/%Y"}).column_types == {
        "a": "datetime:%d/%m/%Y"
    }


def test_overrides_reach_profile_qc_eda_and_pipeline():
    from mlkit.runner import build_default_pipeline

    df = _override_frame()
    types = {"stars": "categorical", "amount": "datetime:excel", "notes": "ignore"}
    opts = ProfileOptions(overrides=types)
    prof = profile_dataframe(df.drop(columns="target"), options=opts)
    assert [prof.columns[c].semantic_type for c in types] == ["categorical", "datetime", "ignored"]

    eda = run_eda(df, target="target", column_types=types)
    assert eda.profile.columns["amount"].semantic_type == "datetime"
    assert "notes" not in set(eda.target_association["feature"])
    assert eda.target_association.iloc[0]["feature"] == "stars"
    assert eda.target_association.iloc[0]["method"] == "cramers_v"  # treated as categorical

    cfg = TaskConfig(
        target="target", column_types=types, cv_folds=3, modelling={"candidates": ["linear"]}
    )
    pipe = build_default_pipeline(cfg)
    X, y = df.drop(columns="target"), df["target"]
    pipe.fit(X, y, cfg=cfg)
    tc = pipe.named_steps["type_caster"]
    assert tc.drop_reasons_["notes"] == "ignored"
    assert (
        tc.inferred_dtypes_["stars"] == "category"
        and tc.inferred_dtypes_["amount"] == "datetime64[ns]"
    )
    feats = pipe.transform(X).columns
    assert any(c.startswith("stars_") for c in feats)  # one-hot, not a number
    assert any(c.startswith("amount_") for c in feats)  # calendar parts from the Excel serial


def test_cli_type_flag(tmp_path):
    from mlkit.cli import cmd_eda

    data = tmp_path / "d.csv"
    _override_frame().to_csv(data, index=False)
    assert cmd_eda(str(data), target="target", column_types={"stars": "categorical"}) == 0


# ============================================================================== fuzzing
_NAMES = ["x", "id", "date", "amount", "Thành phố", "notes", 0, "0", "created_at", "code"]
_KINDS = [
    "float",
    "float_inf_nan",
    "int",
    "huge_int",
    "bool_none",
    "small_cat",
    "missing_tokens",
    "numeric_text",
    "leading_zero",
    "date_text",
    "tz_dates",
    "mixed_tz",
    "timedelta",
    "nullable_int",
    "nullable_str",
    "category_dtype",
    "lists",
    "mixed_objects",
    "all_nan",
    "constant",
    "emoji",
    "long_text",
    "yyyymmdd",
]


def _column(kind: str, n: int, rng: np.random.Generator) -> pd.Series | list | np.ndarray:
    if kind == "float":
        return rng.normal(size=n) * 10.0 ** rng.integers(-3, 6)
    if kind == "float_inf_nan":
        v = rng.normal(size=n)
        v[rng.uniform(size=n) < 0.2] = np.nan
        v[rng.uniform(size=n) < 0.05] = np.inf
        return v
    if kind == "int":
        return rng.integers(-5, 50, n)
    if kind == "huge_int":
        return np.full(n, 2**62, dtype=np.int64) + rng.integers(0, 3, n)
    if kind == "bool_none":
        return pd.Series(rng.choice([True, False, None], n), dtype=object)
    if kind == "small_cat":
        return rng.choice(["a", "b", "c", "d"], n)
    if kind == "missing_tokens":
        return rng.choice(["x", "y", "n/a", "NULL", "-", ""], n)
    if kind == "numeric_text":
        return [f"${v:,.2f}" if i % 7 else "n/a" for i, v in enumerate(rng.uniform(0, 1e6, n))]
    if kind == "leading_zero":
        return [f"{v:05d}" for v in rng.integers(0, 999, n)]
    if kind == "date_text":
        return (
            pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 2000, n), unit="D")
        ).strftime("%d/%m/%Y")
    if kind == "tz_dates":
        return pd.Series(pd.date_range("2024-01-01", periods=n, freq="h", tz="Asia/Ho_Chi_Minh"))
    if kind == "mixed_tz":
        return rng.choice(["2024-01-01T10:00:00+07:00", "2024-03-02T10:00:00Z", "bad"], n)
    if kind == "timedelta":
        return pd.Series(pd.to_timedelta(rng.integers(0, 10_000, n), unit="s"))
    if kind == "nullable_int":
        return pd.array(
            np.where(rng.uniform(size=n) < 0.2, None, rng.integers(0, 9, n)), dtype="Int64"
        )
    if kind == "nullable_str":
        return pd.array(rng.choice(["u", "v", None], n), dtype="string")
    if kind == "category_dtype":
        return pd.Categorical(rng.choice(["p", "q"], n), categories=["p", "q", "r"])
    if kind == "lists":
        return pd.Series([[int(v)] for v in rng.integers(0, 3, n)], dtype=object)
    if kind == "mixed_objects":
        return pd.Series(
            [
                int(v) if v % 3 == 0 else (f"s{v}" if v % 3 == 1 else None)
                for v in rng.integers(0, 30, n)
            ],
            dtype=object,
        )
    if kind == "all_nan":
        return np.full(n, np.nan)
    if kind == "constant":
        return np.full(n, "k", dtype=object)
    if kind == "emoji":
        return rng.choice(["😀", "🔥", "ok", "Hà Nội"], n)
    if kind == "long_text":
        return [
            " ".join(rng.choice(["good", "bad", "slow", "fast"], int(k)))
            for k in rng.integers(4, 30, n)
        ]
    if kind == "yyyymmdd":
        return rng.choice([20240101, 20240215, 20231231, 20230705], n)
    raise AssertionError(kind)


def _target(kind: str, n: int, rng: np.random.Generator) -> pd.Series | None:
    if kind == "none":
        return None
    if kind == "binary":
        return pd.Series(rng.integers(0, 2, n))
    if kind == "binary_text":
        return pd.Series(rng.choice(["yes", "no"], n))
    if kind == "multiclass":
        return pd.Series(rng.choice(["a", "b", "c"], n))
    if kind == "regression":
        return pd.Series(rng.normal(size=n) * 100)
    if kind == "with_nan":
        return pd.Series(np.where(rng.uniform(size=n) < 0.3, np.nan, rng.integers(0, 2, n)))
    return pd.Series(np.zeros(n))  # constant


def _frame(spec: list[tuple], n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = [
        pd.Series(_column(kind, n, rng)).reset_index(drop=True).rename(name) for name, kind in spec
    ]
    return pd.concat(cols, axis=1) if cols else pd.DataFrame(index=range(n))


specs = st.lists(
    st.tuples(st.sampled_from(_NAMES), st.sampled_from(_KINDS)), min_size=1, max_size=6
)
targets = st.sampled_from(
    ["none", "binary", "binary_text", "multiclass", "regression", "with_nan", "constant"]
)
N_FUZZ = int(os.environ.get("MLKIT_FUZZ_EXAMPLES", "60"))  # raise locally for a deeper search
SETTINGS = settings(
    max_examples=N_FUZZ,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@SETTINGS
@given(spec=specs, n=st.integers(0, 120), seed=st.integers(0, 10_000), tkind=targets)
def test_fuzz_profile_and_qc_never_raise(spec, n, seed, tkind):
    df = _frame(spec, n, seed)
    y = _target(tkind, n, np.random.default_rng(seed + 1))
    prof = profile_dataframe(df, y=y)
    assert all(p.semantic_type in SEMANTIC_TYPES for p in prof.columns.values())
    qc = run_qc(df, y=y, enforce_gate=False)
    assert 0.0 <= qc.data_quality_score_ <= 100.0


@settings(
    max_examples=max(10, N_FUZZ // 3),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(spec=specs, n=st.integers(0, 80), seed=st.integers(0, 10_000), tkind=targets)
def test_fuzz_eda_and_charts_never_raise(spec, n, seed, tkind):
    df = _frame(spec, n, seed)
    y = _target(tkind, n, np.random.default_rng(seed + 1))
    if y is not None:
        df = df.assign(__target__=y.to_numpy())
    report = run_eda(df, target="__target__" if y is not None else None)
    for _, fig in report.figures(max_distributions=8, max_target_plots=2):
        plt.close(fig)
    report.to_html()


def _chain():
    from mlkit.features import DatetimeFeatures, TextFeatures
    from mlkit.preprocessing import (
        CategoricalEncoder,
        MissingImputer,
        RareCategoryGrouper,
        TypeCaster,
    )

    return [
        TypeCaster(),
        DatetimeFeatures(),
        TextFeatures(),
        MissingImputer(add_indicator="auto", categorical_strategy="missing_value"),
        RareCategoryGrouper(min_freq=None, min_count=5, max_cardinality=10),
        CategoricalEncoder(),
    ]


def _assert_model_ready(out: pd.DataFrame) -> None:
    for c in out.columns:
        assert pd.api.types.is_numeric_dtype(out[c]), (c, out[c].dtype)
    values = out.to_numpy(dtype=float) if len(out.columns) else np.zeros((len(out), 0))
    assert np.isfinite(values).all()


@SETTINGS
@given(spec=specs, n=st.integers(1, 120), seed=st.integers(0, 10_000), tkind=targets)
def test_fuzz_preprocessing_is_model_ready_and_schema_stable(spec, n, seed, tkind):
    train = _frame(spec, n, seed)
    y = _target(tkind, n, np.random.default_rng(seed + 1))
    y_fit = None if y is None or y.isna().any() or y.nunique() < 2 else y
    steps = _chain()
    X = train
    for step in steps:
        X = step.fit_transform(X, y_fit)
    _assert_model_ready(X)

    batch = _frame(
        spec, 37, seed + 99
    )  # same schema, new values (unseen categories, other dates...)
    Xb = batch
    for step in steps:
        Xb = step.transform(Xb)
    _assert_model_ready(Xb)
    assert list(Xb.columns) == list(X.columns)


# ===================================================== findings from the real-data (OpenML) suite
def test_leakage_through_missingness_is_flagged_but_strong_legit_features_are_not():
    from sklearn.datasets import load_breast_cancer

    rng = np.random.default_rng(11)
    n = 1500
    y = pd.Series(rng.integers(0, 2, n))
    # recorded only after the outcome (like Titanic's lifeboat number): present mostly for y == 1
    boat = pd.Series(
        np.where((y == 1) & (rng.uniform(size=n) < 0.9), rng.integers(1, 20, n), np.nan)
    )
    qc = run_qc(pd.DataFrame({"boat": boat, "x": rng.normal(size=n)}), y=y, enforce_gate=False)
    item = next(i for i in qc.report_.items if i.name == "leakage_suspect")
    assert item.status == "warn" and item.details["missing_signal"][0][0] == "boat"

    bc = load_breast_cancer(as_frame=True)
    qc_bc = run_qc(bc.data, y=bc.target, enforce_gate=False)
    assert not any(
        i.name in ("leakage_suspect", "data_leakage") and i.status != "ok"
        for i in qc_bc.report_.items
    )


def test_special_characters_in_category_values_reach_lightgbm():
    from mlkit.modelling import LGBMTrainer
    from mlkit.preprocessing import CategoricalEncoder

    rng = np.random.default_rng(12)
    n = 400
    df = pd.DataFrame(
        {"range": rng.choice(["[0-1]", "{2,3}", 'a"b', "x:y"], n), "x": rng.normal(size=n)}
    )
    y = rng.integers(0, 2, n)
    out = CategoricalEncoder().fit_transform(df, y)
    assert not any(ch in "".join(out.columns) for ch in '[]{}",:')
    model = LGBMTrainer(n_estimators=10).fit(out, y)  # would raise LightGBMError before
    assert model.predict_proba(out).shape == (n, 2)
    assert len(model.feature_importances_) == out.shape[1]
