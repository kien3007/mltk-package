"""Unit tests for mltk-monitoring."""

import numpy as np
import pandas as pd
import pytest

from mltk_monitoring import (
    DataDriftMonitor,
    PerformanceMonitor,
    PredictionDriftMonitor,
)


def test_data_drift_monitor_numeric():
    np.random.seed(42)
    ref_df = pd.DataFrame({"num": np.random.normal(0, 1, 1000)})

    # Current dataset 1: drawn from same distribution -> no drift
    curr_clean = pd.DataFrame({"num": np.random.normal(0, 1, 500)})

    # Current dataset 2: mean shifted significantly -> large drift
    curr_drift = pd.DataFrame({"num": np.random.normal(3, 1, 500)})

    monitor = DataDriftMonitor()
    monitor.fit(ref_df)

    rep_clean = monitor.evaluate(curr_clean)
    assert rep_clean.passed is True

    rep_drift = monitor.evaluate(curr_drift)
    assert rep_drift.has_warnings is True or rep_drift.has_errors is True


def test_data_drift_monitor_categorical():
    ref_df = pd.DataFrame({"cat": ["A"] * 80 + ["B"] * 20})
    curr_df = pd.DataFrame({"cat": ["A"] * 10 + ["B"] * 90})  # Flipped proportions

    monitor = DataDriftMonitor()
    monitor.fit(ref_df)
    rep = monitor.evaluate(curr_df)
    assert rep.has_warnings is True or rep.has_errors is True


def test_prediction_drift_monitor():
    ref_preds = np.random.beta(2, 5, size=1000)
    curr_preds_stable = np.random.beta(2, 5, size=500)
    curr_preds_drift = np.random.beta(5, 2, size=500)

    p_mon = PredictionDriftMonitor(ref_preds, psi_threshold=0.2)

    rep1 = p_mon.evaluate(curr_preds_stable)
    assert rep1.results[0].passed is True

    rep2 = p_mon.evaluate(curr_preds_drift)
    assert rep2.results[0].passed is False


def test_performance_monitor():
    baseline = {"roc_auc": 0.85, "accuracy": 0.80}
    current_good = {"roc_auc": 0.84, "accuracy": 0.81}
    current_degraded = {"roc_auc": 0.70, "accuracy": 0.65}

    pm = PerformanceMonitor(baseline, degradation_threshold=0.1)

    rep_good = pm.evaluate(current_good)
    assert rep_good.passed is True

    rep_bad = pm.evaluate(current_degraded)
    assert rep_bad.has_warnings is True
