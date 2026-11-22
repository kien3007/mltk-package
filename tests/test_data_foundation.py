"""Data foundation: profiling, QC, preprocessing, EDA and EDA charts across column/target types."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlkit.core import TaskConfig
from mlkit.qc_eda import (
    correlation_ratio,
    cramers_v,
    infer_column,
    profile_dataframe,
    run_eda,
    run_qc,
    univariate_signal,
)
from mlkit.qc_eda.profile import coerce_numeric, name_looks_like_id

rng = np.random.default_rng(42)
N = 400


def messy_frame(n: int = N) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "row_no": np.arange(n),
            "price_txt": [f"${v:,.2f}" for v in rng.uniform(10, 5000, n)],
            "pct": [f"{v:.1f}%" for v in rng.uniform(0, 100, n)],
            "eu_decimal": [f"{v:.2f}".replace(".", ",") for v in rng.normal(size=n)],
            "active": rng.choice(["Yes", "No"], n),
            "flag": rng.choice([True, False], n),
            "signup": pd.date_range("2023-01-01", periods=n, freq="D").strftime("%d/%m/%Y"),
            "city": rng.choice([f"city_{i}" for i in range(300)], n),
            "segment": rng.choice(["A", "B", "C"], n),
            "review": [
                " ".join(rng.choice(["good", "bad", "fast", "slow", "price"], 8)) for _ in range(n)
            ],
            "mixed": [str(v) if i % 3 else "n/a" for i, v in enumerate(rng.integers(0, 100, n))],
            "const": 7,
            "empty": np.nan,
            "income": np.where(rng.uniform(size=n) < 0.2, np.nan, rng.lognormal(10, 1, n)),
            "with_inf": np.where(rng.uniform(size=n) < 0.02, np.inf, rng.normal(size=n)),
            "amount_paid": rng.integers(0, 10000, n),
        }
    )
    df["income_dup"] = df["income"]
    return df


EXPECTED_TYPES = {
    "customer_id": "identifier",
    "row_no": "identifier",
    "price_txt": "numeric",
    "pct": "numeric",
    "eu_decimal": "numeric",
    "active": "boolean",
    "flag": "boolean",
    "signup": "datetime",
    "city": "high_cardinality",
    "segment": "categorical",
    "review": "text",
    "mixed": "numeric",
    "const": "constant",
    "empty": "empty",
    "income": "numeric",
    "with_inf": "numeric",
    "amount_paid": "numeric",
}


# ------------------------------------------------------------------------- profile
class TestProfile:
    def test_semantic_types(self):
        prof = profile_dataframe(messy_frame())
        got = {c: prof.columns[c].semantic_type for c in EXPECTED_TYPES}
        assert got == EXPECTED_TYPES
        assert prof.duplicate_columns == [["income", "income_dup"]]
        assert any("disguised missing" in n for n in prof.columns["mixed"].notes)
        assert prof.columns["with_inf"].numeric["n_inf"] > 0

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("1,234.5", 1234.5),
            ("$1,000", 1000.0),
            ("12%", 12.0),
            ("3,25", 3.25),
            ("-7", -7.0),
            ("abc", np.nan),
        ],
    )
    def test_coerce_numeric(self, raw, expected):
        got = coerce_numeric(pd.Series([raw])).iloc[0]
        assert (np.isnan(got) and np.isnan(expected)) or got == pytest.approx(expected)

    @pytest.mark.parametrize(
        "name, is_id",
        [
            ("customer_id", True),
            ("customerID", True),
            ("CustomerId", True),
            ("household_key", True),
            ("amount_paid", False),
            ("valid", False),
            ("idle_time", False),
            ("zip_code", False),
        ],
    )
    def test_id_names(self, name, is_id):
        assert name_looks_like_id(name) is is_id

    def test_robust_to_odd_inputs(self):
        df = pd.DataFrame({0: [[1], [2], [1]] * 20, 1: rng.normal(size=60), "s": ["a"] * 60})
        prof = profile_dataframe(df)
        assert (
            set(prof.columns) == {"0", "1", "s"} and prof.columns["s"].semantic_type == "constant"
        )
        assert profile_dataframe(pd.DataFrame()).n_cols == 0

    def test_target_profiles(self):
        X = pd.DataFrame({"x": rng.normal(size=200)})
        assert profile_dataframe(X, y=pd.Series(rng.integers(0, 2, 200))).target.kind == "binary"
        assert (
            profile_dataframe(X, y=pd.Series(rng.choice(list("abc"), 200))).target.kind
            == "multiclass"
        )
        assert profile_dataframe(X, y=pd.Series(rng.normal(size=200))).target.kind == "regression"
        t = profile_dataframe(X, y=pd.Series([0] * 190 + [1] * 10)).target
        assert t.minority_share == pytest.approx(0.05)


# ------------------------------------------------------------------------------ QC
def _names(qc, status=None):
    return {i.name for i in qc.report_.items if status is None or i.status == status}


class TestQC:
    def test_messy_findings(self):
        qc = run_qc(messy_frame(), y=pd.Series(rng.integers(0, 2, N)), enforce_gate=False)
        warns = _names(qc, "warn")
        for check in (
            "identifier_columns",
            "text_columns",
            "disguised_missing",
            "duplicate_columns",
            "infinite_values",
            "constant_columns",
            "empty_columns",
            "high_cardinality",
        ):
            assert check in warns, check
        assert "data_leakage" in _names(qc, "ok")
        issues = qc.issues_frame_
        assert {"customer_id", "income_dup"} <= set(issues["column"])
        assert qc.profile_.n_rows == N

    @pytest.mark.parametrize("kind", ["categorical_copy", "numeric_binary", "regression"])
    def test_leakage_detected_for_any_type_pair(self, kind):
        n = 500
        if kind == "categorical_copy":
            y = pd.Series(rng.choice(["yes", "no"], n))
            X = pd.DataFrame(
                {"status_code": y.map({"yes": "S1", "no": "S0"}), "x": rng.normal(size=n)}
            )
            cfg = TaskConfig(target="y")
        elif kind == "numeric_binary":
            x = rng.normal(size=n)
            y = pd.Series((x > 0.3).astype(int))
            X = pd.DataFrame({"score": x, "noise": rng.normal(size=n)})
            cfg = TaskConfig(target="y")
        else:
            x = rng.normal(size=n)
            y = pd.Series(np.exp(x))  # monotone transform -> perfect rank correlation
            X = pd.DataFrame({"x": x, "noise": rng.normal(size=n)})
            cfg = TaskConfig(target="y", task_type="regression")
        qc = run_qc(X, y=y, cfg=cfg, enforce_gate=False)
        item = next(i for i in qc.report_.items if i.name == "data_leakage")
        assert item.status == "error"
        assert len(item.details["leakage_columns"]) == 1

    def test_no_false_leakage_on_strong_but_noisy_or_tiny_data(self):
        n = 2000
        x = rng.normal(size=n)
        y = pd.Series((x + rng.normal(scale=0.3, size=n) > 0).astype(int))  # AUC ~0.97, not a leak
        assert "data_leakage" in _names(
            run_qc(pd.DataFrame({"x": x}), y=y, enforce_gate=False), "ok"
        )
        tiny = pd.DataFrame({"x": [0, 1] * 10})
        assert univariate_signal(tiny["x"], pd.Series([0, 1] * 10), "classification") is None

    def test_target_checks(self):
        X = pd.DataFrame({"x": rng.normal(size=300)})
        qc = run_qc(
            X, y=pd.Series([0] * 297 + [1] * 3), cfg=TaskConfig(target="y"), enforce_gate=False
        )
        assert "rare_classes" in _names(qc, "error")
        qc = run_qc(
            X, y=pd.Series([0] * 290 + [1] * 10), cfg=TaskConfig(target="y"), enforce_gate=False
        )
        assert "class_imbalance" in _names(qc, "warn")
        with pytest.raises(Exception, match="QC Quality Gate"):
            run_qc(X, y=pd.Series([0] * 297 + [1] * 3), cfg=TaskConfig(target="y"))


# ------------------------------------------------------------------- preprocessing
class TestPreprocessing:
    def test_type_caster_converts_and_drops(self):
        from mlkit.preprocessing import TypeCaster

        df = messy_frame()
        tc = TypeCaster().fit(df)
        out = tc.transform(df)
        assert set(tc.drop_reasons_) >= {"customer_id", "row_no", "const", "empty", "income_dup"}
        assert tc.drop_reasons_["income_dup"].startswith("duplicate")
        assert out["price_txt"].dtype == float and out["price_txt"].between(10, 5000).all()
        assert out["pct"].max() <= 100 and out["eu_decimal"].dtype == float
        assert set(out["active"].unique()) == {0.0, 1.0} and set(out["flag"].unique()) == {0.0, 1.0}
        assert pd.api.types.is_datetime64_any_dtype(out["signup"])
        assert out["mixed"].isna().sum() == (df["mixed"] == "n/a").sum()
        assert np.isfinite(out["with_inf"].dropna()).all()
        assert out["city"].dtype == "category" and out["review"].dtype == object

        batch = df.head(5).copy()
        batch.loc[0, "price_txt"] = "n/a"
        batch.loc[1, "amount_paid"] = np.inf
        b = tc.transform(batch)
        assert np.isnan(b.loc[0, "price_txt"]) and np.isnan(b.loc[1, "amount_paid"])

    def test_forced_dtype_and_datetime_time_col(self):
        from mlkit.preprocessing import TypeCaster

        df = pd.DataFrame(
            {"zip": [10001, 10002, 10003] * 20, "when": pd.date_range("2024-01-01", periods=60)}
        )
        out = TypeCaster(dtypes={"zip": "category"}).fit_transform(
            df, cfg=TaskConfig(target="y", time_col="when")
        )
        assert out["zip"].dtype == "category"
        assert "when" in out.columns  # real dates are kept for DatetimeFeatures

    def test_text_features(self):
        from mlkit.features import TextFeatures

        df = messy_frame()[["review", "segment"]]
        out = TextFeatures().fit_transform(df)
        assert "review" not in out.columns and "segment" in out.columns
        assert (out["review_n_words"] == 8).all()

    def test_rare_grouper_respects_cardinality_and_missing(self):
        from mlkit.preprocessing import RareCategoryGrouper

        df = pd.DataFrame(
            {
                "low": ["a"] * 50 + ["b"] * 45 + ["c"] * 3 + [None] * 2,
                "high": [f"v{i % 60}" for i in range(100)],
            }
        )
        out = RareCategoryGrouper(min_freq=None, min_count=10, max_cardinality=10).fit_transform(df)
        assert set(out["low"].dropna()) == {"a", "b", "__rare__"} and out["low"].isna().sum() == 2
        assert out["high"].nunique() == 60  # high-cardinality column untouched

    def test_encoder_strategies(self):
        from mlkit.preprocessing import CategoricalEncoder

        df = pd.DataFrame(
            {
                "mid": [f"m{i % 20}" for i in range(400)],
                "grade": rng.choice(["B", "A", "C", "D", "E", "F", "G", "H", "I", "J", "K"], 400),
            }
        )
        y = rng.integers(0, 2, 400)
        enc = CategoricalEncoder(strategies={"grade": "ordinal"}).fit(df, y)
        assert enc.encoding_plan_ == {"mid": "target", "grade": "ordinal"}
        assert enc.ordinal_maps_["grade"]["A"] == 0  # sorted, deterministic order
        unsup = CategoricalEncoder().fit(df)
        assert unsup.encoding_plan_["mid"] == "frequency"
        out = unsup.transform(pd.DataFrame({"mid": ["m1", "unseen"], "grade": ["A", "A"]}))
        assert out["mid"].tolist() == [pytest.approx(0.05), 0.0]
        with pytest.raises(ValueError, match="strategy"):
            CategoricalEncoder(strategies={"mid": "magic"}).fit(df, y)

    def test_imputer_auto_indicators(self):
        from mlkit.preprocessing import MissingImputer

        df = pd.DataFrame(
            {"often": [np.nan] * 10 + list(range(90)), "rare": [np.nan] + list(range(999))[:99]}
        )
        imp = MissingImputer(add_indicator="auto", indicator_min_rate=0.05).fit(df)
        assert imp.indicator_cols_ == ["often"]


# ----------------------------------------------------------------------------- EDA
class TestEDA:
    def test_association_measures(self):
        a = pd.Series(rng.choice(list("xyz"), 3000))
        assert cramers_v(a, a) == pytest.approx(1.0, abs=1e-3)
        assert cramers_v(a, pd.Series(rng.choice(list("pq"), 3000))) < 0.05
        v = pd.Series(rng.normal(size=3000))
        assert correlation_ratio(a, a.map({"x": 0.0, "y": 5.0, "z": 10.0})) == pytest.approx(1.0)
        assert correlation_ratio(a, v) < 0.06

    @pytest.mark.parametrize("target_kind", ["binary", "multiclass", "regression", None])
    def test_run_eda_any_target(self, target_kind, tmp_path):
        df = messy_frame()
        city_effect = df["city"].str.len()
        if target_kind == "binary":
            df["t"] = (df["segment"] == "A").astype(int)
        elif target_kind == "multiclass":
            df["t"] = df["segment"].map({"A": "low", "B": "mid", "C": "high"})
        elif target_kind == "regression":
            df["t"] = df["eu_decimal"].str.replace(",", ".").astype(float) * 3 + city_effect
        report = run_eda(df, target="t" if target_kind else None)
        assert report.target_kind == target_kind
        if target_kind:
            top = report.target_association.iloc[0]
            assert top["feature"] in ("segment", "eu_decimal") and top["strength"] > 0.8
            assert report.target_by_feature and not report.missingness.empty
        assert report.associations.shape[0] == report.associations.shape[1] > 1
        assert "Rows: 400" in report.summary()
        html = report.to_html(tmp_path / "eda.html")
        assert (tmp_path / "eda.html").exists() and "Data quality" in html and "<img" in html
        for _, fig in report.figures():
            plt.close(fig)

    def test_sampling(self):
        df = pd.DataFrame({"x": rng.normal(size=5000), "t": rng.integers(0, 2, 5000)})
        assert run_eda(df, target="t", sample_rows=1000, with_qc=False).profile.n_rows == 1000

    def test_charts_handle_degenerate_columns(self):
        from mlkit.viz import plot_column_overview, plot_distributions

        df = pd.DataFrame(
            {"allnan": [np.nan] * 50, "x": rng.normal(size=50), "c": ["a"] * 25 + ["b"] * 25}
        )
        prof = profile_dataframe(df)
        for fig in (
            plot_column_overview(prof),
            plot_distributions(df, prof, columns=["allnan", "x", "c"]),
        ):
            assert hasattr(fig, "savefig")
            plt.close(fig)


def test_cli_eda(tmp_path):
    from mlkit.cli import cmd_eda

    data = tmp_path / "d.csv"
    df = messy_frame()
    df["t"] = rng.integers(0, 2, len(df))
    df.to_csv(data, index=False)
    assert (
        cmd_eda(
            str(data),
            target="t",
            out_path=str(tmp_path / "r.html"),
            profile_json=str(tmp_path / "p.json"),
        )
        == 0
    )
    payload = json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))
    assert payload["profile"]["columns"]["price_txt"]["coercion"] == "numeric"
    assert payload["issues"]
    assert cmd_eda(str(data), target="nope") == 1


def test_infer_column_unnamed_series():
    assert infer_column(pd.Series([1.5, 2.5, 3.5] * 30)).semantic_type == "numeric"
