"""End-to-end integration test across all MLTK packages."""

import tempfile
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from mltk_features import DatetimeFeatures, RatioFeatures
from mltk_modelling import LGBMTrainer, load_artifact, save_artifact
from mltk_monitoring import DataDriftMonitor
from mltk_preprocessing import (
    CategoricalEncoder,
    MissingImputer,
    OutlierClipper,
    TypeCaster,
)
from mltk_qc import run_qc
from mltk_selection import CorrelationFilter, ModelImportanceSelector


def test_full_mltk_pipeline_integration():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame(
        {
            "tenure": np.random.uniform(1, 60, size=n),
            "monthly_charges": np.random.uniform(20, 100, size=n),
            "contract": np.random.choice(["Month-to-month", "One year"], size=n),
            "signup_date": np.random.choice(["2024-01-01", "2024-06-15", "2024-12-01"], size=n),
        }
    )
    y = pd.Series((df["monthly_charges"] > 50).astype(int), name="churn")

    # 1. QC
    qc_report = run_qc(df, target=y)
    assert qc_report.passed is True

    # 2. Pipeline assembly
    pipe = Pipeline(
        [
            ("type_caster", TypeCaster()),
            ("imputer", MissingImputer()),
            ("clipper", OutlierClipper()),
            ("ratio", RatioFeatures(pairs=[("monthly_charges", "tenure")])),
            ("datetime", DatetimeFeatures(columns=["signup_date"])),
            ("encoder", CategoricalEncoder(low_card_threshold=5)),
            ("corr_filter", CorrelationFilter(threshold=0.99)),
            ("importance_selector", ModelImportanceSelector(top_k=5, random_state=42)),
            ("model", LGBMTrainer(n_estimators=10, random_state=42)),
        ]
    )

    # 3. Fit
    pipe.fit(df, y)
    preds = pipe.predict(df)
    assert len(preds) == n

    # 4. Save artifact
    with tempfile.TemporaryDirectory() as tmpdir:
        art_dir = save_artifact(
            pipeline=pipe,
            X_train=df,
            y_train=y,
            metrics={"acc": 0.9},
            output_dir=tmpdir,
            model_name="integration_test_model",
        )

        # 5. Load artifact
        loaded_pipe, meta = load_artifact(art_dir)
        loaded_preds = loaded_pipe.predict(df)
        np.testing.assert_array_equal(preds, loaded_preds)

        # 6. Drift Monitor from artifact
        monitor = DataDriftMonitor.from_artifact(art_dir)
        df_new = df.copy()
        drift_report = monitor.evaluate(df_new)
        assert drift_report.passed is True
