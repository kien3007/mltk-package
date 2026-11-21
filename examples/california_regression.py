"""End-to-End Regression Pipeline on the California Housing Dataset.

Demonstrates:
1. QC & EDA Gatekeeper inspection
2. Preprocessing with IQR clipping and robust scaling
3. Feature Engineering with ratio features (AveRooms / AveBedrms) and binning
4. Feature Selection with variance and correlation filtering
5. Model Arena: Multi-model CV benchmark (Dummy vs Ridge vs RF vs LightGBM)
6. Monitoring: Baseline snapshot and PSI drift checking
7. Production Bundle export and verification
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from mlkit.core.bundle import load_bundle
from mlkit.core.config import TaskConfig
from mlkit.core.pipeline import Pipeline
from mlkit.features import BinningFeatures, RatioFeatures
from mlkit.modelling import ModelArena
from mlkit.monitoring import DriftMonitor
from mlkit.preprocessing import FeatureScaler, MissingImputer, OutlierClipper, TypeCaster
from mlkit.qc_eda import QualityCheckStep
from mlkit.selection import CorrelationFilter, VarianceFilter


def main() -> None:
    config_file = Path("examples/housing_config.yaml")
    cfg = TaskConfig.from_yaml(config_file)
    print(f"Loaded config: task_type={cfg.task_type}, target={cfg.target}")

    # 1. Load Data
    data_path = Path("examples/data/california_housing.csv")
    df = pd.read_csv(data_path)
    print(f"Loaded California Housing dataset: {df.shape}")

    y = df[cfg.target]
    X = df.drop(columns=[cfg.target])

    # 2. QC Stage
    print("\n--- [Step 1] QC & EDA Inspection ---")
    qc = QualityCheckStep(enforce_gate=cfg.quality_gate)
    qc.fit(X, y=y, cfg=cfg)
    print(qc.report_.summary())

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # 4. Pipeline Assembly
    pipeline = Pipeline(
        [
            ("type_caster", TypeCaster()),
            ("imputer", MissingImputer()),
            ("clipper", OutlierClipper(factor=2.0)),
            ("scaler", FeatureScaler(method="robust")),
            ("ratio", RatioFeatures(pairs=[("AveRooms", "AveBedrms"), ("Population", "AveOccup")])),
            ("binning", BinningFeatures(columns=["HouseAge"], n_bins=5)),
            ("variance_filter", VarianceFilter(threshold=0.0)),
            ("corr_filter", CorrelationFilter(threshold=0.98)),
            ("modelling_arena", ModelArena(cv=cfg.cv_folds, enforce_baseline_gate=True, random_state=cfg.seed)),
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
    test_rmse = (mean_squared_error(y_test, y_pred)) ** 0.5
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print(f"\nTest RMSE: {test_rmse:.4f}")
    print(f"Test MAE:  {test_mae:.4f}")
    print(f"Test R2:   {test_r2:.4f}")

    # 7. Production Bundle Export
    bundle_path = Path("artifacts/housing_bundle")
    pipeline.save(
        path=bundle_path,
        cfg=cfg,
        X_sample=X_train.head(10),
        metrics={"test_rmse": test_rmse, "test_r2": test_r2},
    )
    print(f"\n--- [Step 6] Saved Production Bundle to: {bundle_path} ---")

    # Verify reload
    loaded_pipe, loaded_cfg, metadata = load_bundle(bundle_path)
    reloaded_preds = loaded_pipe.predict(X_test)
    assert (abs(y_pred - reloaded_preds) < 1e-5).all(), "Loaded pipeline predictions do not match!"
    print("✅ Verified: Reloaded bundle predictions match 100%!")

    # 8. Drift Monitoring
    monitor = DriftMonitor()
    monitor.fit(X_train)
    drift_rep = monitor.monitor(X_test)
    print("\nDrift Monitoring Report on Test Batch:")
    print(drift_rep.summary())


if __name__ == "__main__":
    main()
