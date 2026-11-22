"""End-to-End Classification Pipeline on the Adult Census Dataset.

Demonstrates:
1. QC & EDA Gatekeeper inspection
2. Preprocessing with missing imputation, rare grouping, adaptive categorical encoding, and scaling
3. Feature Engineering
4. Feature Selection
5. Model Arena: Multi-model CV benchmark (Dummy vs Linear vs RF vs LightGBM) with baseline gate
6. Monitoring: Baseline snapshot and PSI drift checking
7. Production Bundle export and verification
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from mlkit.core.bundle import load_bundle
from mlkit.core.config import TaskConfig
from mlkit.core.pipeline import Pipeline
from mlkit.features import RatioFeatures
from mlkit.modelling import ModelArena
from mlkit.monitoring import DriftMonitor
from mlkit.preprocessing import (
    CategoricalEncoder,
    FeatureScaler,
    MissingImputer,
    RareCategoryGrouper,
    TypeCaster,
)
from mlkit.qc_eda import QualityCheckStep
from mlkit.selection import CorrelationFilter, ModelImportanceSelector, VarianceFilter


def main() -> None:
    config_file = Path("examples/adult_config.yaml")
    cfg = TaskConfig.from_yaml(config_file)
    print(f"Loaded config: task_type={cfg.task_type}, target={cfg.target}")

    # 1. Load Data
    data_path = Path("examples/data/adult.csv")
    df = pd.read_csv(data_path)
    print(f"Loaded Adult dataset: {df.shape}")

    # Standardize target to binary 0/1
    y_raw = df[cfg.target].astype(str).str.strip().str.replace(".", "", regex=False)
    y = (y_raw == ">50K").astype(int)
    X = df.drop(columns=[cfg.target])

    # 2. QC Stage
    print("\n--- [Step 1] QC & EDA Inspection ---")
    qc = QualityCheckStep(enforce_gate=cfg.quality_gate)
    qc.fit(X, y=y, cfg=cfg)
    print(qc.report_.summary())

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # 4. Pipeline Assembly
    pipeline = Pipeline(
        [
            ("type_caster", TypeCaster()),
            ("imputer", MissingImputer(numeric_strategy="median", categorical_strategy="mode")),
            ("rare_grouper", RareCategoryGrouper(min_freq=0.01)),
            ("encoder", CategoricalEncoder(low_card_threshold=10, high_card_threshold=50)),
            ("scaler", FeatureScaler(method="standard")),
            ("ratio", RatioFeatures(pairs=[("capital-gain", "capital-loss")])),
            ("variance_filter", VarianceFilter(threshold=0.0)),
            ("corr_filter", CorrelationFilter(threshold=0.95)),
            ("importance_selector", ModelImportanceSelector(top_k=25)),
            (
                "modelling_arena",
                ModelArena(cv=cfg.cv_folds, enforce_baseline_gate=True, random_state=cfg.seed),
            ),
        ]
    )

    # 5. Fit & Model Arena
    print("\n--- [Steps 2-5] Fitting Pipeline & Benchmarking Model Zoo ---")
    pipeline.fit(X_train, y_train, cfg=cfg)

    arena = pipeline.named_steps["modelling_arena"]
    print("\nModel Arena Comparison:")
    print(arena.comparison_table_.to_string(index=False))
    print(f"🏆 Champion Model: {arena.best_model_name_}")

    # 6. Test Set Evaluation
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob

    test_auc = roc_auc_score(y_test, prob_pos)
    print(f"\nTest ROC-AUC: {test_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))

    # 7. Production Bundle Export
    bundle_path = Path("artifacts/adult_bundle")
    pipeline.save(
        path=bundle_path,
        cfg=cfg,
        X_sample=X_train.head(10),
        metrics={"test_roc_auc": test_auc},
    )
    print(f"\n--- [Step 6] Saved Production Bundle to: {bundle_path} ---")

    # Verify reload
    loaded_pipe, loaded_cfg, metadata = load_bundle(bundle_path)
    reloaded_preds = loaded_pipe.predict(X_test)
    assert (y_pred == reloaded_preds).all(), "Loaded pipeline predictions do not match!"
    print("✅ Verified: Reloaded bundle predictions match 100%!")

    # 8. Drift Monitoring
    monitor = DriftMonitor()
    monitor.fit(X_train)
    drift_rep = monitor.monitor(X_test)
    print("\nDrift Monitoring Report on Test Batch:")
    print(drift_rep.summary())


if __name__ == "__main__":
    main()
