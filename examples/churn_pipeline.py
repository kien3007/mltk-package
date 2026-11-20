"""End-to-end example: Customer Churn Model using MLTK."""

from __future__ import annotations

import tempfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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


def create_sample_churn_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """Generate a realistic synthetic tabular dataset with numeric, categorical, and datetime columns."""
    np.random.seed(42)
    tenure = np.random.exponential(scale=20, size=n_samples).clip(1, 72)
    monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)
    total_charges = tenure * monthly_charges + np.random.normal(0, 50, size=n_samples)

    # Inject missing values
    missing_mask = np.random.rand(n_samples) < 0.05
    tenure[missing_mask] = np.nan

    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.20])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_samples,
    )
    signup_dates = pd.date_range("2024-01-01", periods=365).astype(str)
    signup_date = np.random.choice(signup_dates, size=n_samples)

    # Churn probability based on contract, charges, and tenure
    churn_logit = (
        -1.5
        + (monthly_charges / 50.0)
        - (np.nan_to_num(tenure, nan=10) / 20.0)
        + 1.2 * (contract == "Month-to-month")
    )
    churn_prob = 1.0 / (1.0 + np.exp(-churn_logit))
    churn = (np.random.rand(n_samples) < churn_prob).astype(int)

    return pd.DataFrame(
        {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract": contract,
            "payment_method": payment_method,
            "signup_date": signup_date,
            "churn": churn,
        }
    )


def run_pipeline() -> None:
    print("=== Step 1: Load Data & Inspect ===")
    df = create_sample_churn_dataset(n_samples=600)
    print(f"Dataset shape: {df.shape}")

    # Explicit convention: separate X and y
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\n=== Step 2: Quality Control & Validation ===")
    qc_report = run_qc(X_train, target=y_train)
    print(f"QC passed: {qc_report.passed}")
    print(f"QC total checks: {len(qc_report.results)}")
    qc_report.raise_if_failed()

    print("\n=== Step 3: Define Assembly Pipeline ===")
    pipe = Pipeline(
        [
            ("type_caster", TypeCaster()),
            (
                "imputer",
                MissingImputer(numeric_strategy="median", categorical_strategy="most_frequent"),
            ),
            ("clipper", OutlierClipper(lower_quantile=0.01, upper_quantile=0.99)),
            ("ratio", RatioFeatures(pairs=[("total_charges", "monthly_charges")])),
            ("datetime", DatetimeFeatures(columns=["signup_date"])),
            ("encoder", CategoricalEncoder(low_card_threshold=5, high_card_threshold=20, cv=3)),
            ("corr_filter", CorrelationFilter(threshold=0.98)),
            ("importance_selector", ModelImportanceSelector(top_k=8, random_state=42)),
            ("model", LGBMTrainer(n_estimators=30, learning_rate=0.05, random_state=42)),
        ]
    )

    print("\n=== Step 4: Fit Pipeline ===")
    pipe.fit(X_train, y_train)

    test_preds = pipe.predict(X_test)
    test_probs = pipe.predict_proba(X_test)[:, 1]
    from sklearn.metrics import roc_auc_score
    test_auc = float(roc_auc_score(y_test, test_probs))
    print(f"Pipeline fitted successfully! Test AUC: {test_auc:.4f}")

    print("\n=== Step 5: Save Complete Artifact Bundle ===")
    with tempfile.TemporaryDirectory() as tmp_artifact_dir:
        art_path = save_artifact(
            pipeline=pipe,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            metrics={"test_auc": test_auc},
            output_dir=tmp_artifact_dir,
            model_name="churn_predictor_v1",
            random_state=42,
        )
        print(f"Artifact saved at: {art_path}")

        print("\n=== Step 6: Load Artifact & Verify ===")
        loaded_pipe, metadata = load_artifact(art_path)
        loaded_preds = loaded_pipe.predict(X_test)
        np.testing.assert_array_equal(test_preds, loaded_preds)
        print("Loaded pipeline predictions verified!")

        print("\n=== Step 7: Data Drift Monitoring on New Incoming Batch ===")
        monitor = DataDriftMonitor.from_artifact(art_path)
        # Create new batch with simulated distribution shift
        X_new_batch = X_test.copy()
        X_new_batch["monthly_charges"] = X_new_batch["monthly_charges"] * 1.5

        drift_report = monitor.evaluate(X_new_batch)
        print(f"Drift evaluation completed! Passed: {drift_report.passed}")
        for r in drift_report.results:
            if not r.passed:
                print(f"  Alert: [{r.name}] {r.message}")

    print("\n=== MLTK Workflow Completed Successfully! ===")


if __name__ == "__main__":
    run_pipeline()
