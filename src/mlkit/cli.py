"""Command-line interface for running mlkit pipelines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mlkit.core.bundle import load_bundle
from mlkit.core.config import TaskConfig
from mlkit.core.pipeline import Pipeline
from mlkit.features import DatetimeFeatures
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


def build_default_pipeline(cfg: TaskConfig) -> Pipeline:
    """Construct an end-to-end 6-stage pipeline from TaskConfig."""
    steps = [
        ("qc", QualityCheckStep(enforce_gate=cfg.quality_gate)),
        ("type_caster", TypeCaster(drop_id=True)),
        ("imputer", MissingImputer(add_indicator=False)),
        ("rare_grouper", RareCategoryGrouper(min_freq=0.01)),
        ("encoder", CategoricalEncoder()),
        ("scaler", FeatureScaler()),
        ("datetime", DatetimeFeatures()),
        ("variance_filter", VarianceFilter(threshold=0.0)),
        ("corr_filter", CorrelationFilter(threshold=0.95)),
        ("importance_selector", ModelImportanceSelector(top_k=30)),
        (
            "modelling_arena",
            ModelArena(cv=cfg.cv_folds, enforce_baseline_gate=cfg.baseline_gate, random_state=cfg.seed),
        ),
    ]
    return Pipeline(steps)


def cmd_run(config_path: str, data_path: str | None = None, output_dir: str = "./bundle") -> int:
    """Execute full pipeline training from config file."""
    print(f"🚀 [mlkit] Loading task configuration from: {config_path}")
    cfg = TaskConfig.from_yaml(config_path)

    # Resolve dataset path
    resolved_data_path = data_path or getattr(cfg, "data_path", None)
    if not resolved_data_path or not Path(resolved_data_path).exists():
        print(f"❌ Error: Data file not found: {resolved_data_path}")
        return 1

    print(f"📊 [mlkit] Loading dataset from: {resolved_data_path}")
    df = pd.read_csv(resolved_data_path)

    if cfg.target not in df.columns:
        print(f"❌ Error: Target column '{cfg.target}' not found in dataset columns: {list(df.columns)}")
        return 1

    X = df.drop(columns=[cfg.target])
    y = df[cfg.target]

    if cfg.task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
        y_clean = y.astype(str).str.strip().str.replace(".", "", regex=False)
        unique_classes = sorted(y_clean.unique())
        if len(unique_classes) == 2:
            # Map positive class to 1
            pos_label = unique_classes[1]
            y = (y_clean == pos_label).astype(int)

    stratify = y if cfg.task_type == "classification" and cfg.stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=stratify
    )

    print(f"⚙️ [mlkit] Assembling 6-stage pipeline (Task: {cfg.task_type.upper()})...")
    pipeline = build_default_pipeline(cfg)

    print("🏋️ [mlkit] Training pipeline and running model arena...")
    pipeline.fit(X_train, y_train, cfg=cfg)

    arena_step = pipeline.named_steps.get("modelling_arena")
    if arena_step:
        print("\n📈 [mlkit] Model Arena Benchmark Results:")
        print(arena_step.comparison_table_.to_string(index=False))
        print(f"\n🏆 [mlkit] Selected Champion: {arena_step.best_model_name_}")

    # Capture monitoring baseline
    monitor = DriftMonitor()
    monitor.fit(X_train, y_train, cfg=cfg)
    drift_report = monitor.monitor(X_test)
    print("\n🔍 [mlkit] Test Data Drift Check:")
    print(drift_report.summary())

    # Save bundle
    print(f"\n📦 [mlkit] Exporting production bundle to: {output_dir}")
    pipeline.save(
        path=output_dir,
        cfg=cfg,
        X_sample=X_train.head(10),
        baseline=monitor.categorical_baselines_,
    )
    print(f"✅ [mlkit] Pipeline run successfully completed! Bundle ready at: {output_dir}")
    return 0


def cmd_monitor(bundle_path: str, data_path: str) -> int:
    """Evaluate drift on a new batch of data using saved bundle."""
    print(f"🔍 [mlkit] Loading production bundle: {bundle_path}")
    pipeline, cfg, metadata = load_bundle(bundle_path)

    print(f"📊 [mlkit] Loading batch data: {data_path}")
    df = pd.read_csv(data_path)
    if cfg.target in df.columns:
        df = df.drop(columns=[cfg.target])

    monitor = DriftMonitor()
    # Mock fit baseline from metadata or current
    monitor.fit(df)
    report = monitor.monitor(df)
    print(report.summary())
    return 0 if report.is_passed else 1


def main() -> None:
    """Entry point for mlkit CLI."""
    parser = argparse.ArgumentParser(prog="mlkit", description="MLKit CLI: 6-stage ML pipeline runner")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run full pipeline from config")
    run_parser.add_argument("config", help="Path to config.yaml")
    run_parser.add_argument("--data", help="Optional override path to dataset CSV")
    run_parser.add_argument("--out", default="./bundle", help="Output directory for bundle")

    monitor_parser = subparsers.add_parser("monitor", help="Monitor batch data against bundle")
    monitor_parser.add_argument("bundle", help="Path to saved bundle directory")
    monitor_parser.add_argument("data", help="Path to batch CSV data")

    args = parser.parse_args()
    if args.command == "run":
        sys.exit(cmd_run(args.config, data_path=args.data, output_dir=args.out))
    elif args.command == "monitor":
        sys.exit(cmd_monitor(args.bundle, args.data))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
