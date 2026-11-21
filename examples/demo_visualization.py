"""
Comprehensive Demonstration of MLKit's Extended 15-Chart Visualization & Tabbed Dashboard Suite.

Demonstrates:
- 1. Data Quality & Profiling: Missing rates, target imbalance, correlation heatmap, target distribution overlay.
- 2. Model Arena & Learning Curves: Multi-model benchmark, training/val sample efficiency curve.
- 3. Business Decision Economics: ROC curve, Precision-Recall curve, Confusion matrix, Cumulative Gains, Decile Lift.
- 4. Explainable AI & Calibration: Feature importance, SHAP global attribution, Calibration reliability curve.
- 5. Drift Monitoring: Population Stability Index (PSI) drift tracking.
- 6. Interactive Tabbed HTML Dashboard: Standalone multi-tab executive report.
"""

from __future__ import annotations

import re
from pathlib import Path

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
    plot_cumulative_gains,
    plot_drift_psi,
    plot_feature_distribution_by_target,
    plot_feature_importance,
    plot_learning_curve,
    plot_lift_curve,
    plot_missing_rates,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_shap_summary,
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
    print("=" * 75)
    print("🎨 MLKIT 15-CHART ADVANCED VISUALIZATION & TABBED DASHBOARD SUITE")
    print("=" * 75)

    banking_dir = Path(r"C:\Users\kienn\Downloads\banking")
    output_dir = Path("reports/viz_demo")
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("\n📂 [1/6] Loading Banking Data (Transaction Decline Detection)...")
    tx_df = pd.read_excel(banking_dir / "transactions_data.xlsx", nrows=25000)
    cards_df = pd.read_excel(banking_dir / "cards_data.xlsx")
    users_df = pd.read_excel(banking_dir / "users_data.xlsx", nrows=1000)

    tx_df["is_declined"] = tx_df["errors"].notna().astype(int)
    tx_df["amount"] = tx_df["amount"].apply(clean_currency)
    cards_df["credit_limit"] = cards_df["credit_limit"].apply(clean_currency)
    users_df["yearly_income"] = users_df["yearly_income"].apply(clean_currency)
    users_df["total_debt"] = users_df["total_debt"].apply(clean_currency)

    merged = tx_df.merge(
        cards_df[["id", "card_brand", "card_type", "credit_limit", "has_chip"]],
        left_on="card_id",
        right_on="id",
        how="left",
        suffixes=("", "_card"),
    )
    merged = merged.merge(
        users_df[["id", "current_age", "yearly_income", "total_debt", "credit_score"]],
        left_on="client_id",
        right_on="id",
        how="left",
        suffixes=("", "_user"),
    )

    drop_cols = [
        "id",
        "id_card",
        "id_user",
        "client_id",
        "card_id",
        "merchant_city",
        "merchant_state",
        "zip",
        "errors",
        "merchant_id",
    ]
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

    # 2. Stage 1: Data Quality & Exploratory Profiling
    print("\n🔍 [2/6] Generating Data Quality & Exploratory Profiling Charts...")
    qc = run_qc(X, y=y)

    fig_missing = plot_missing_rates(X, threshold=0.1)
    fig_missing.savefig(fig_dir / "01_missing_rates.png")
    print("  ✅ [01/15] Missing Rate Chart")

    fig_target = plot_target_distribution(y, task_type="classification")
    fig_target.savefig(fig_dir / "02_target_distribution.png")
    print("  ✅ [02/15] Target Distribution (Imbalance) Chart")

    num_cols = X.select_dtypes(include=[np.number])
    fig_corr = plot_correlation_heatmap(num_cols, top_n=12)
    fig_corr.savefig(fig_dir / "03_correlation_heatmap.png")
    print("  ✅ [03/15] Correlation Heatmap")

    fig_density = plot_feature_distribution_by_target(df, feature_col="amount", target_col="is_declined")
    fig_density.savefig(fig_dir / "04_amount_distribution_by_target.png")
    print("  ✅ [04/15] Feature Density by Target (KDE Overlay)")

    # 3. Stage 2: Pipeline Training & Arena
    print("\n⚙️ [3/6] Fitting MLKit Pipeline and Benchmarking Models...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

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

    fig_arena = plot_arena_comparison(arena_step)
    fig_arena.savefig(fig_dir / "05_arena_comparison.png")
    print("  ✅ [05/15] Model Arena Multi-Model Benchmark Chart")

    # Learning curve on sample of processed data
    X_tr_proc = pipeline.transform(X_train.head(1500))
    y_tr_sub = y_train.head(1500)
    fig_learn = plot_learning_curve(arena_step.best_estimator_._estimator, X_tr_proc, y_tr_sub, cv=3)
    fig_learn.savefig(fig_dir / "06_learning_curve.png")
    print("  ✅ [06/15] Sample Efficiency Learning Curve")

    # 4. Stage 3: Business Decision Economics
    print("\n🎯 [4/6] Generating Business Decision & Economics Curves...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    fig_cm = plot_confusion_matrix(y_test, y_pred, labels=["Success (0)", "Declined (1)"])
    fig_cm.savefig(fig_dir / "07_confusion_matrix.png")
    print("  ✅ [07/15] Confusion Matrix (Normalized)")

    fig_roc = plot_roc_curve(y_test, y_prob)
    fig_roc.savefig(fig_dir / "08_roc_curve.png")
    print("  ✅ [08/15] ROC Curve & AUC Score")

    fig_pr = plot_precision_recall_curve(y_test, y_prob)
    fig_pr.savefig(fig_dir / "09_precision_recall_curve.png")
    print("  ✅ [09/15] Precision-Recall Curve (AP Score)")

    fig_gains = plot_cumulative_gains(y_test, y_prob)
    fig_gains.savefig(fig_dir / "10_cumulative_gains.png")
    print("  ✅ [10/15] Cumulative Gains Chart (Deciles)")

    fig_lift = plot_lift_curve(y_test, y_prob, n_deciles=10)
    fig_lift.savefig(fig_dir / "11_lift_curve.png")
    print("  ✅ [11/15] Decile Lift Multiplier Curve")

    # 5. Stage 4: Explainability & Calibration
    print("\n🧠 [5/6] Generating Explainability (SHAP) & Calibration Charts...")
    fig_importance = plot_feature_importance(arena_step, top_n=12)
    fig_importance.savefig(fig_dir / "12_feature_importance.png")
    print("  ✅ [12/15] Standard Feature Importance")

    fig_calib = plot_calibration_curve(y_test, y_prob, n_bins=8)
    fig_calib.savefig(fig_dir / "13_calibration_curve.png")
    print("  ✅ [13/15] Probability Calibration Reliability Diagram")

    fig_shap = plot_shap_summary(arena_step.best_estimator_, X_tr_proc, top_n=10)
    fig_shap.savefig(fig_dir / "14_shap_summary.png")
    print("  ✅ [14/15] SHAP Global Feature Impact Chart")

    # 6. Stage 5: Monitoring & Tabbed HTML Dashboard
    print("\n📡 [6/6] Generating Drift Monitoring & Standalone Tabbed Dashboard...")
    monitor = DriftMonitor()
    monitor.fit(X_train)
    drift_rep = monitor.monitor(X_test)

    fig_drift = plot_drift_psi(drift_rep, warn_threshold=0.05, error_threshold=0.1)
    fig_drift.savefig(fig_dir / "15_drift_psi.png")
    print("  ✅ [15/15] Population Stability Index (PSI) Drift Chart")

    # Compile all 15 figures into tabbed HTML dashboard
    html_file = output_dir / "mlkit_diagnostic_dashboard.html"
    generate_html_report(
        reports=[qc.report_, pipeline.report(), drift_rep],
        title="MLKit Comprehensive Diagnostic Dashboard — Transaction Decline Detection",
        summary_stats={
            "Analyzed Dataset": f"{len(df):,} transactions",
            "Champion Algorithm": arena_step.best_model_name_,
            "Data Quality Score": f"{qc.data_quality_score_} / 100",
            "Target Prevalence": f"{y.mean()*100:.2f}% positive",
        },
        figures=[
            ("1. Missing Rate per Column", fig_missing),
            ("2. Target Class Distribution", fig_target),
            ("3. Feature Correlation Heatmap", fig_corr),
            ("4. Amount Density Distribution by Target Class", fig_density),
            ("5. Model Arena Benchmark (ROC-AUC)", fig_arena),
            ("6. Sample Efficiency Learning Curve", fig_learn),
            ("7. Confusion Matrix (Normalized)", fig_cm),
            ("8. ROC Curve & AUC Score", fig_roc),
            ("9. Precision-Recall Curve (AP Score)", fig_pr),
            ("10. Cumulative Gains Chart (Deciles)", fig_gains),
            ("11. Decile Lift Multiplier Curve", fig_lift),
            ("12. Standard Feature Importance", fig_importance),
            ("13. Probability Calibration Curve", fig_calib),
            ("14. SHAP Feature Attribution Impact", fig_shap),
            ("15. Population Stability Index (PSI) Drift", fig_drift),
        ],
        output_path=html_file,
    )

    print("\n" + "=" * 75)
    print(f"🎉 SUCCESS: All 15 charts generated in: {fig_dir.resolve()}")
    print(f"🌐 Tabbed HTML Dashboard created at: {html_file.resolve()}")
    print("=" * 75)


if __name__ == "__main__":
    main()
