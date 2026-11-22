"""Comprehensive unit tests covering all 6 stages of MLKit."""

import numpy as np
import pandas as pd
import pytest

from mlkit.core.errors import BaselineComparisonError, QualityGateError
from mlkit.features import BinningFeatures, DatetimeFeatures, GroupAggFeatures, RatioFeatures
from mlkit.modelling import (
    DummyTrainer,
    LGBMTrainer,
    LinearTrainer,
    ModelArena,
    RandomForestTrainer,
)
from mlkit.monitoring import DriftMonitor, calculate_psi
from mlkit.preprocessing import (
    CategoricalEncoder,
    FeatureScaler,
    MissingImputer,
    RareCategoryGrouper,
    TypeCaster,
)
from mlkit.qc_eda import QualityCheckStep


# --- 1. QC & EDA Tests ---
def test_qc_empty_dataframe():
    qc = QualityCheckStep(enforce_gate=False)
    qc.fit(pd.DataFrame())
    assert qc.report_.has_errors
    assert any(i.name == "empty_data" for i in qc.report_.items)


def test_qc_single_class_target_error():
    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    y = pd.Series([1, 1, 1, 1])
    qc = QualityCheckStep(enforce_gate=True)
    with pytest.raises(QualityGateError, match="Target contains only 1 unique class"):
        qc.fit(df, y=y)


def test_qc_data_leakage_detected():
    df = pd.DataFrame({"feat": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 5})
    y = df["feat"] * 2.0  # Perfect linear correlation!
    qc = QualityCheckStep(enforce_gate=False, leakage_threshold=0.99)
    qc.fit(df, y=y)
    assert any(i.name == "data_leakage" and i.status == "error" for i in qc.report_.items)


# --- 2. Preprocessing Tests ---
def test_preprocessing_pipeline_flow():
    df = pd.DataFrame(
        {
            "id": ["ID1", "ID2", "ID3", "ID4", "ID5", "ID6"],
            "age": [25.0, np.nan, 35.0, 40.0, 120.0, 28.0],
            "city": ["Hanoi", "Hanoi", "Saigon", "Danang", "RarePlace", np.nan],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1])

    # 1. TypeCaster drops ID
    tc = TypeCaster()
    df_tc = tc.fit_transform(df)
    assert "id" not in df_tc.columns

    # 2. MissingImputer with flag
    imp = MissingImputer(add_indicator=True)
    df_imp = imp.fit_transform(df_tc)
    assert not df_imp["age"].isna().any()
    assert "age_is_missing" in df_imp.columns

    # 3. Rare grouper
    rg = RareCategoryGrouper(min_freq=0.3)
    df_rg = rg.fit_transform(df_imp)
    assert "__rare__" in df_rg["city"].values

    # 4. Categorical encoder
    enc = CategoricalEncoder(low_card_threshold=5)
    df_enc = enc.fit_transform(df_rg, y=y)
    assert "city" not in df_enc.columns  # One-hot encoded!

    # 5. Scaler
    scaler = FeatureScaler(method="standard")
    df_scaled = scaler.fit_transform(df_enc)
    assert isinstance(df_scaled, pd.DataFrame)
    assert "age" in df_scaled.columns


# --- 3. Feature Engineering Tests ---
def test_feature_engineering_transformers():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5, freq="D"),
            "income": [100.0, 200.0, 150.0, 300.0, 250.0],
            "debt": [20.0, 50.0, 30.0, 60.0, 50.0],
            "grp": ["A", "A", "B", "B", "A"],
        }
    )

    # Datetime features
    dt = DatetimeFeatures(columns=["date"])
    df_dt = dt.fit_transform(df)
    assert "date_year" in df_dt.columns
    assert "date_month" in df_dt.columns

    # Ratio features
    rf = RatioFeatures(pairs=[("debt", "income")])
    df_rf = rf.fit_transform(df_dt)
    assert "debt_div_income" in df_rf.columns
    assert "debt_sub_income" in df_rf.columns

    # Binning features
    bf = BinningFeatures(columns=["income"], n_bins=3)
    df_bf = bf.fit_transform(df_rf)
    assert "income_bin" in df_bf.columns

    # Group agg features
    gaf = GroupAggFeatures(group_cols=["grp"], target_cols=["income"], aggregations=["mean"])
    df_gaf = gaf.fit_transform(df_bf)
    assert "grp_income_mean" in df_gaf.columns


# --- 4. Modelling & Arena Tests ---
def test_modelling_trainers_and_arena():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
    y = (X["f1"] + X["f2"] > 0).astype(int)

    # Dummy
    dum = DummyTrainer(task_type="classification")
    dum.fit(X, y)
    assert len(dum.predict(X)) == n
    assert dum.predict_proba(X).shape == (n, 2)

    # Linear
    lin = LinearTrainer(task_type="classification")
    lin.fit(X, y)
    assert len(lin.predict(X)) == n
    assert len(lin.feature_importances_) == 2

    # Random Forest
    rf = RandomForestTrainer(n_estimators=10, task_type="classification")
    rf.fit(X, y)
    assert len(rf.predict(X)) == n

    # LightGBM
    lgb = LGBMTrainer(n_estimators=10, task_type="classification")
    lgb.fit(X, y)
    assert len(lgb.predict(X)) == n

    # Model Arena (runs zoo and selects best model)
    arena = ModelArena(cv=3, random_state=42)
    arena.fit(X, y)
    assert arena.best_model_name_ != ""
    assert not arena.comparison_table_.empty
    preds = arena.predict(X)
    assert len(preds) == n


def test_arena_baseline_gate_failure():
    # Model that fails to beat baseline
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0]})
    y = [1, 1, 1, 1]  # degenerate
    arena = ModelArena(cv=2, enforce_baseline_gate=True)
    with pytest.raises((BaselineComparisonError, ValueError)):
        arena.fit(X, y)


# --- 5. Monitoring & PSI Tests ---
def test_exact_psi_calculation():
    # Exact identical distributions MUST give PSI == 0.0
    arr1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    psi_identical = calculate_psi(arr1, arr1)
    assert psi_identical == 0.0

    # Shifted distribution gives PSI > 0
    arr2 = arr1 + 10.0
    psi_shifted = calculate_psi(arr1, arr2)
    assert psi_shifted > 0.1


def test_drift_monitor_inspection():
    np.random.seed(42)
    base_df = pd.DataFrame(
        {
            "num": np.random.normal(0, 1, 100),
            "cat": np.random.choice(["A", "B"], size=100),
        }
    )
    monitor = DriftMonitor()
    monitor.fit(base_df)

    # Identical current data -> stable report
    curr_df = base_df.copy()
    report = monitor.monitor(curr_df)
    assert report.is_passed
    assert all(i.status == "ok" for i in report.items)

    # Heavily shifted data -> warning/error report
    shifted_df = pd.DataFrame(
        {
            "num": np.random.normal(10, 1, 100),
            "cat": ["B"] * 100,
        }
    )
    drift_report = monitor.monitor(shifted_df)
    assert drift_report.has_warnings or drift_report.has_errors
