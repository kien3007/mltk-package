"""Tests for upgrades: MultiTableAggregator, Bootstrap CI, Calibration, HTML Reports, and DQS."""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Test Data
# ---------------------------------------------------------------------------
rng = np.random.default_rng(123)
N = 100

primary_df = pd.DataFrame({
    "account_id": [f"acc_{i}" for i in range(N)],
    "age": rng.integers(18, 65, size=N),
    "balance": rng.normal(10_000, 2_000, size=N),
    "target": rng.choice([0, 1], size=N),
})

# 1-to-many transactions table: 5 transactions per account
trans_records = []
for acc in primary_df["account_id"]:
    n_trans = rng.integers(1, 10)
    for _t in range(n_trans):
        trans_records.append({
            "account_id": acc,
            "amount": float(rng.uniform(10, 500)),
            "trans_type": rng.choice(["credit", "debit"]),
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 180))),
        })
secondary_df = pd.DataFrame(trans_records)


# ===========================================================================
# 1. MultiTableAggregator Tests
# ===========================================================================
class TestMultiTableAggregator:
    def test_basic_aggregation(self):
        from mlkit.features.engineering import MultiTableAggregator

        aggregator = MultiTableAggregator(
            secondary_df=secondary_df,
            join_key="account_id",
            numeric_aggs=["mean", "sum", "count"],
            categorical_aggs=["nunique"],
            prefix="tx_",
        )
        aggregator.fit(primary_df)
        transformed = aggregator.transform(primary_df)

        assert "tx_amount_mean" in transformed.columns
        assert "tx_amount_sum" in transformed.columns
        assert "tx_amount_count" in transformed.columns
        assert "tx_trans_type_nunique" in transformed.columns
        assert len(transformed) == len(primary_df)

    def test_windowed_time_aggregation(self):
        from mlkit.features.engineering import MultiTableAggregator

        aggregator = MultiTableAggregator(
            secondary_df=secondary_df,
            join_key="account_id",
            time_col="date",
            window_days=[30, 90],
            prefix="tx_",
        )
        aggregator.fit(primary_df)
        transformed = aggregator.transform(primary_df)

        assert any("last_30d" in c for c in transformed.columns)
        assert any("last_90d" in c for c in transformed.columns)

    def test_invalid_key_warning(self):
        from mlkit.features.engineering import MultiTableAggregator

        aggregator = MultiTableAggregator(
            secondary_df=secondary_df,
            join_key="non_existent_key",
        )
        aggregator.fit(primary_df)
        assert aggregator.report_.has_warnings


# ===========================================================================
# 2. Bootstrap CI Tests
# ===========================================================================
class TestBootstrapCI:
    def test_classification_ci(self):
        from mlkit.modelling.evaluation import compute_bootstrap_ci

        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1] * 10)
        y_pred = y_true.copy()  # perfect prediction
        y_prob = np.column_stack([1.0 - y_pred, y_pred.astype(float)])

        ci_res = compute_bootstrap_ci(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            task_type="classification",
            n_bootstraps=50,
            random_state=42,
        )

        assert "accuracy" in ci_res
        assert "f1" in ci_res
        assert "roc_auc" in ci_res
        assert ci_res["accuracy"]["mean"] > 0.95
        assert ci_res["accuracy"]["ci_lower"] <= ci_res["accuracy"]["ci_upper"]

    def test_regression_ci(self):
        from mlkit.modelling.evaluation import compute_bootstrap_ci

        y_true = rng.normal(100, 10, size=100)
        y_pred = y_true + rng.normal(0, 1, size=100)

        ci_res = compute_bootstrap_ci(
            y_true=y_true,
            y_pred=y_pred,
            task_type="regression",
            n_bootstraps=50,
            random_state=42,
        )

        assert "rmse" in ci_res
        assert "r2" in ci_res
        assert ci_res["rmse"]["ci_lower"] <= ci_res["rmse"]["ci_upper"]


# ===========================================================================
# 3. Calibration Tests
# ===========================================================================
class TestCalibration:
    def test_calibrated_model_wrapper(self):
        from sklearn.linear_model import LogisticRegression

        from mlkit.modelling.evaluation import CalibratedModelWrapper

        X = primary_df[["age", "balance"]]
        y = primary_df["target"]

        base = LogisticRegression()
        calibrated = CalibratedModelWrapper(base_estimator=base, method="sigmoid", cv=2)
        calibrated.fit(X, y)

        probs = calibrated.predict_proba(X)
        preds = calibrated.predict(X)

        assert probs.shape == (len(X), 2)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert len(preds) == len(X)

    def test_calibration_curve_plot(self):
        import matplotlib.pyplot as plt

        from mlkit.viz import plot_calibration_curve

        y_true = rng.choice([0, 1], size=100)
        y_prob = rng.uniform(0, 1, size=100)

        fig = plot_calibration_curve(y_true, y_prob, n_bins=5)
        assert hasattr(fig, "savefig")
        plt.close(fig)


# ===========================================================================
# 4. HTML Report Generation Tests
# ===========================================================================
class TestHTMLReport:
    def test_report_to_html(self, tmp_path):
        from mlkit.core.report import Report

        rep = Report("Test Report")
        rep.add("schema_check", "ok", "Schema is valid.", {"cols": 10})
        rep.add("outlier_check", "warn", "Found 3 outliers.")

        html_file = tmp_path / "test_report.html"
        html = rep.to_html(output_path=html_file)

        assert "<!DOCTYPE html>" in html
        assert "Test Report" in html
        assert "schema_check" in html
        assert html_file.exists()
        assert len(html_file.read_text(encoding="utf-8")) > 200

    def test_pipeline_to_html_report(self, tmp_path):
        import matplotlib.pyplot as plt

        from mlkit.core.pipeline import Pipeline
        from mlkit.qc_eda.qc import QualityCheckStep

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])

        pipe = Pipeline(steps=[("qc", QualityCheckStep())])
        pipe.fit(primary_df[["age", "balance"]], y=primary_df["target"])

        html_file = tmp_path / "pipeline_report.html"
        html = pipe.to_html_report(
            output_path=html_file,
            summary_stats={"Dataset Size": f"{len(primary_df)} rows"},
            figures=[("Test Plot", fig)],
        )
        plt.close(fig)

        assert "Pipeline Execution & Diagnostic Report" in html
        assert "data:image/png;base64," in html
        assert "Dataset Size" in html
        assert html_file.exists()


# ===========================================================================
# 5. Data Quality Score Tests
# ===========================================================================
class TestDataQualityScore:
    def test_clean_data_high_score(self):
        from mlkit.qc_eda.qc import QualityCheckStep

        qc = QualityCheckStep()
        qc.fit(primary_df[["age", "balance"]], y=primary_df["target"])

        assert hasattr(qc, "data_quality_score_")
        assert qc.data_quality_score_ >= 80.0
        assert any(it.name == "data_quality_score" for it in qc.report_.items)

    def test_dirty_data_penalized_score(self):
        from mlkit.qc_eda.qc import QualityCheckStep

        dirty_df = primary_df.copy()
        # Inject 50% missing values in balance
        dirty_df.loc[:50, "balance"] = np.nan
        # Add 30 duplicate rows
        dirty_df = pd.concat([dirty_df, dirty_df.iloc[:30]], ignore_index=True)

        qc = QualityCheckStep(enforce_gate=False)
        qc.fit(dirty_df[["age", "balance"]], y=dirty_df["target"])

        assert qc.data_quality_score_ < 80.0
