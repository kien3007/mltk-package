"""
Comprehensive Demonstration of MLKit's Visualization & Dashboard Suite.

Generates diagnostic charts for every pipeline stage and compiles them into
a standalone HTML report dashboard.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mlkit.core.config import TaskConfig
from mlkit.core.pipeline import Pipeline
from mlkit.modelling import ModelArena
from mlkit.monitoring import DriftMonitor
from mlkit.preprocessing import (
    CategoricalEncoder,
    FeatureScaler,
    MissingImputer,
    TypeCaster,
)
from mlkit.qc_eda import run_qc
from mlkit.selection import VarianceFilter
from mlkit.viz import (
    generate_html_report,
    plot_arena_comparison,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_drift_psi,
    plot_feature_importance,
    plot_missing_rates,
    plot_roc_curve,
    plot_target_distribution,
)


def clean_currency(val: any) -> float:
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = re.sub(r"[^\d.-]", "", str(val))
    try:
        return float(s)
    except Exception:
        return 0.0


def main():
    print("=" * 70)
    print("🎨 MLKIT VISUALIZATION SUITE DEMONSTRATION")
    print("=" * 70)

    banking_dir = Path(r"C:\Users\kienn\Downloads\banking")
    output_dir = Path("reports/viz_demo")
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("\n📂 [1/5] Loading Banking Data (Transaction Failure Detection)...")
    tx_df = pd.read_excel(banking_dir / "transactions_data.xlsx", nrows=25000)
    cards_df = pd.read_excel(banking_dir / "cards_data.xlsx")
    users_df = pd.read_excel(banking_dir / "users_data.xlsx", nrows=1000)

    # Prepare Transaction Failure target: 1 if error/declined, 0 if successful
    tx_df["is_declined"] = tx_df["errors"].notna().astype(int)
    tx_df["amount"] = tx_df["amount"].apply(clean_currency)
    cards_df["credit_limit"] = cards_df["credit_limit"].apply(clean_currency)
    users_df["yearly_income"] = users_df["yearly_income"].apply(clean_currency)
    users_df["total_debt"] = users_df["total_debt"].apply(clean_currency)

    # Join 1-1 with cards and users
    merged = tx_df.merge(cards_df[["id", "card_brand", "card_type", "credit_limit", "has_chip"]],
                         left_on="card_id", right_on="id", how="left", suffixes=("", "_card"))
    merged = merged.merge(users_df[["id", "current_age", "yearly_income", "total_debt", "credit_score"]],
                          left_on="client_id", right_on="id", how="left", suffixes=("", "_user"))

    # Feature selection for modeling
    drop_cols = ["id", "id_card", "id_user", "client_id", "card_id", "merchant_city",
                 "merchant_state", "zip", "errors", "merchant_id"]
    df = merged.drop(columns=[c for c in drop_cols if c in merged.columns])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["hour"] = df["date"].dt.hour
    df["dayofweek"] = df["date"].dt.dayofweek
    df = df.drop(columns=["date"])

    target = "is_declined"
    X = df.drop(columns=[target])
    y = df[target]

    print(f"  • Total samples: {len(df):,} transactions")
    print(f"  • Declined transactions: {y.sum():,} ({y.mean()*100:.2f}%)")

    # 2. Stage 1 Visualizations: QC & EDA
    print("\n🔍 [2/5] Generating QC & EDA Visualizations...")
    qc = run_qc(X, y=y)

    fig_missing = plot_missing_rates(X, threshold=0.1)
    fig_missing.savefig(fig_dir / "01_missing_rates.png")
    print("  ✅ [1/9] Saved: 01_missing_rates.png")

    fig_target = plot_target_distribution(y, task_type="classification")
    fig_target.savefig(fig_dir / "02_target_distribution.png")
    print("  ✅ [2/9] Saved: 02_target_distribution.png")

    num_cols = X.select_dtypes(include=[np.number])
    fig_corr = plot_correlation_heatmap(num_cols, top_n=12)
    fig_corr.savefig(fig_dir / "03_correlation_heatmap.png")
    print("  ✅ [3/9] Saved: 03_correlation_heatmap.png")

    # 3. Train Pipeline & Arena
    print("\n⚙️ [3/5] Fitting MLKit Pipeline and Benchmarking Models...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    cfg = TaskConfig(task_type="classification", target=target, cv_folds=3, baseline_gate=False)
    pipeline = Pipeline([
        ("caster", TypeCaster(drop_id=False)),
        ("imputer", MissingImputer(add_indicator=False)),
        ("encoder", CategoricalEncoder()),
        ("scaler", FeatureScaler()),
        ("var_filter", VarianceFilter(threshold=0.0)),
        ("arena", ModelArena(cv=3, enforce_baseline_gate=False, random_state=42)),
    ])

    pipeline.fit(X_train, y_train, cfg=cfg)
    arena_step = pipeline.named_steps["arena"]

    print("\n🏆 Model Arena Comparison Table:")
    print(arena_step.comparison_table_.to_string(index=False))

    # 4. Stage 2 Visualizations: Modelling & Performance
    print("\n📊 [4/5] Generating Model Evaluation & Diagnostic Visualizations...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    fig_arena = plot_arena_comparison(arena_step)
    fig_arena.savefig(fig_dir / "04_arena_comparison.png")
    print("  ✅ [4/9] Saved: 04_arena_comparison.png")

    fig_importance = plot_feature_importance(arena_step, top_n=12)
    fig_importance.savefig(fig_dir / "05_feature_importance.png")
    print("  ✅ [5/9] Saved: 05_feature_importance.png")

    fig_cm = plot_confusion_matrix(y_test, y_pred, labels=["Success (0)", "Declined (1)"])
    fig_cm.savefig(fig_dir / "06_confusion_matrix.png")
    print("  ✅ [6/9] Saved: 06_confusion_matrix.png")

    fig_roc = plot_roc_curve(y_test, y_prob)
    fig_roc.savefig(fig_dir / "07_roc_curve.png")
    print("  ✅ [7/9] Saved: 07_roc_curve.png")

    fig_calib = plot_calibration_curve(y_test, y_prob, n_bins=8)
    fig_calib.savefig(fig_dir / "08_calibration_curve.png")
    print("  ✅ [8/9] Saved: 08_calibration_curve.png")

    # 5. Stage 3 Visualizations: Drift Monitoring & Dashboard
    print("\n📡 [5/5] Generating Drift Monitoring & Standalone HTML Dashboard...")
    monitor = DriftMonitor()
    monitor.fit(X_train)
    drift_rep = monitor.monitor(X_test)

    fig_drift = plot_drift_psi(drift_rep, warn_threshold=0.05, error_threshold=0.1)
    fig_drift.savefig(fig_dir / "09_drift_psi.png")
    print("  ✅ [9/9] Saved: 09_drift_psi.png")

    # Compile into standalone HTML Report
    html_file = output_dir / "mlkit_diagnostic_dashboard.html"
    generate_html_report(
        reports=[qc.report_, pipeline.report(), drift_rep],
        title="MLKit Diagnostic & Visualization Dashboard — Banking Decline Detection",
        summary_stats={
            "Analyzed Samples": f"{len(df):,} transactions",
            "Champion Model": arena_step.best_model_name_,
            "Data Quality Score": f"{qc.data_quality_score_} / 100",
            "Imbalance Rate": f"{y.mean()*100:.2f}% positive",
        },
        figures=[
            ("1. Missing Rate per Column", fig_missing),
            ("2. Target Class Distribution (Imbalance)", fig_target),
            ("3. Feature Correlation Heatmap", fig_corr),
            ("4. Model Arena Zoo Benchmark", fig_arena),
            ("5. Top Feature Importances", fig_importance),
            ("6. Confusion Matrix (Normalized)", fig_cm),
            ("7. ROC Curve & AUC Score", fig_roc),
            ("8. Probability Calibration Reliability Diagram", fig_calib),
            ("9. Population Stability Index (PSI) Drift", fig_drift),
        ],
        output_path=html_file,
    )

    print("\n" + "=" * 70)
    print(f"🎉 SUCCESS: All 9 charts generated in: {fig_dir.resolve()}")
    print(f"🌐 HTML Dashboard created at: {html_file.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
