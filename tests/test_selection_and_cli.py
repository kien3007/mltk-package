"""Tests for feature selection, CLI runners, and edge-case robustness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlkit.core.config import TaskConfig
from mlkit.core.errors import QualityGateError

# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
N = 100

df_sample = pd.DataFrame({
    "num1": rng.normal(0, 1, size=N),
    "num2": rng.normal(5, 2, size=N),
    "constant": np.zeros(N),  # 0 variance
    "cat": rng.choice(["cat1", "cat2", "cat3"], size=N),
})
# Make num3 strictly collinear with num1
df_sample["collinear"] = df_sample["num1"] * 2.0 + 0.0001 * rng.normal(0, 1, size=N)

y_cls = pd.Series(rng.choice([0, 1], size=N))
y_reg = pd.Series(rng.normal(10, 2, size=N))


# ===========================================================================
# 1. Feature Selection Tests
# ===========================================================================
class TestFeatureSelectors:
    def test_variance_filter(self):
        from mlkit.selection import VarianceFilter

        vf = VarianceFilter(threshold=0.0)
        vf.fit(df_sample)
        out = vf.transform(df_sample)

        assert "constant" not in out.columns
        assert "constant" in vf.dropped_features_
        assert "num1" in out.columns
        assert "cat" in out.columns

    def test_correlation_filter(self):
        from mlkit.selection import CorrelationFilter

        cf = CorrelationFilter(threshold=0.95)
        cf.fit(df_sample)
        out = cf.transform(df_sample)

        assert "collinear" in cf.dropped_features_
        assert "collinear" not in out.columns
        assert "num1" in out.columns

    def test_mutual_info_selector_classification(self):
        from mlkit.selection import MutualInformationSelector

        mi = MutualInformationSelector(top_k=2)
        cfg = TaskConfig(task_type="classification", target="target")
        mi.fit(df_sample[["num1", "num2", "constant"]], y=y_cls, cfg=cfg)
        out = mi.transform(df_sample[["num1", "num2", "constant"]])

        assert len(out.columns) == 2

    def test_mutual_info_selector_regression(self):
        from mlkit.selection import MutualInformationSelector

        mi = MutualInformationSelector(top_k=1)
        cfg = TaskConfig(task_type="regression", target="target")
        mi.fit(df_sample[["num1", "num2"]], y=y_reg, cfg=cfg)
        out = mi.transform(df_sample[["num1", "num2"]])

        assert len(out.columns) == 1

    def test_model_importance_selector(self):
        from mlkit.selection import ModelImportanceSelector

        mis = ModelImportanceSelector(top_k=2)
        cfg = TaskConfig(task_type="classification", target="target")
        mis.fit(df_sample[["num1", "num2", "constant"]], y=y_cls, cfg=cfg)
        out = mis.transform(df_sample[["num1", "num2", "constant"]])

        assert len(out.columns) == 2


# ===========================================================================
# 2. CLI Runners & Default Pipeline Tests
# ===========================================================================
class TestCLI:
    def test_build_default_pipeline(self):
        from mlkit.cli import build_default_pipeline

        cfg = TaskConfig(task_type="classification", target="class")
        pipe = build_default_pipeline(cfg)

        assert len(pipe.steps) == 11
        step_names = [s[0] for s in pipe.steps]
        assert "qc" in step_names
        assert "encoder" in step_names
        assert "modelling_arena" in step_names

    def test_cmd_run_and_monitor(self, tmp_path):
        from mlkit.cli import cmd_monitor, cmd_run

        data_csv = tmp_path / "train_data.csv"
        df = pd.DataFrame({
            "age": rng.integers(20, 60, size=150),
            "income": rng.normal(50000, 10000, size=150),
            "label": rng.choice([0, 1], size=150),
        })
        df.to_csv(data_csv, index=False)

        cfg_file = tmp_path / "task.yaml"
        cfg_file.write_text(f"""
task_type: classification
target: label
data_path: "{data_csv.as_posix()}"
baseline_gate: false
quality_gate: true
""", encoding="utf-8")

        bundle_dir = tmp_path / "bundle_out"
        ret = cmd_run(str(cfg_file), output_dir=str(bundle_dir))
        assert ret == 0
        assert (bundle_dir / "pipeline.joblib").exists()

        # Run monitor CLI
        ret_mon = cmd_monitor(str(bundle_dir), str(data_csv))
        assert ret_mon == 0


# ===========================================================================
# 3. Edge Cases & Robustness Matrix
# ===========================================================================
class TestEdgeCases:
    def test_single_class_target_fails_gate(self):
        from mlkit.qc_eda import QualityCheckStep

        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        single_class_y = pd.Series([1, 1, 1, 1, 1])

        qc = QualityCheckStep(enforce_gate=True)
        with pytest.raises(QualityGateError):
            qc.fit(df, y=single_class_y)

    def test_all_null_column_handled_by_imputer(self):
        from mlkit.preprocessing import MissingImputer

        df = pd.DataFrame({
            "valid": [1.0, 2.0, 3.0, 4.0],
            "all_null": [np.nan, np.nan, np.nan, np.nan],
        })
        imputer = MissingImputer(add_indicator=False)
        imputer.fit(df)
        out = imputer.transform(df)

        assert out["valid"].isna().sum() == 0
        # Imputer fills with 0 if all values are NaN
        assert out["all_null"].isna().sum() == 0

    def test_unseen_category_in_transform(self):
        from mlkit.preprocessing import CategoricalEncoder

        train_df = pd.DataFrame({"cat": ["A", "B", "A", "B"]})
        test_df = pd.DataFrame({"cat": ["A", "C", "D", "B"]})  # C and D are unseen

        enc = CategoricalEncoder()
        enc.fit(train_df)
        out_test = enc.transform(test_df)

        assert len(out_test) == len(test_df)
        # Should not raise exception, transforms cleanly
        assert not out_test.isna().all().all()

    def test_empty_dataframe_handled_by_qc(self):
        from mlkit.qc_eda import QualityCheckStep

        empty_df = pd.DataFrame()
        qc = QualityCheckStep(enforce_gate=False)
        qc.fit(empty_df)

        assert qc.report_.has_errors
        assert any(it.name == "empty_data" for it in qc.report_.items)
