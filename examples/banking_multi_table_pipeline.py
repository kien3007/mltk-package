"""
End-to-End Multi-Table Banking Pipeline Example using MLKit.

Demonstrates:
- Multi-table relational feature aggregation (Users + Cards + Transactions).
- 6-step automated machine learning pipeline with ModelArena benchmark.
- HTML diagnostic dashboard generation with embedded charts.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mlkit.core.bundle import save_bundle
from mlkit.core.config import TaskConfig
from mlkit.core.pipeline import Pipeline
from mlkit.features import MultiTableAggregator
from mlkit.modelling import ModelArena
from mlkit.monitoring import DriftMonitor
from mlkit.preprocessing import (
    CategoricalEncoder,
    FeatureScaler,
    MissingImputer,
    TypeCaster,
)
from mlkit.qc_eda import run_qc
from mlkit.selection import CorrelationFilter, VarianceFilter
from mlkit.viz import (
    generate_html_report,
    plot_arena_comparison,
    plot_correlation_heatmap,
    plot_feature_importance,
)


def clean_currency(val: any) -> float:
    """Parse string currency like '$25,300' into numeric float."""
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
    banking_dir = Path(r"C:\Users\kienn\Downloads\banking")
    if not banking_dir.exists():
        print(f"Directory not found: {banking_dir}")
        return

    print("🚀 [Banking Multi-Table] Loading raw relational tables...")
    users_df = pd.read_excel(banking_dir / "users_data.xlsx", nrows=1000)
    cards_df = pd.read_excel(banking_dir / "cards_data.xlsx")
    tx_df = pd.read_excel(banking_dir / "transactions_data.xlsx", nrows=20000)

    # Clean currency strings in users, cards, and transactions
    for col in ["per_capita_income", "yearly_income", "total_debt"]:
        if col in users_df.columns:
            users_df[col] = users_df[col].apply(clean_currency)

    if "credit_limit" in cards_df.columns:
        cards_df["credit_limit"] = cards_df["credit_limit"].apply(clean_currency)

    if "amount" in tx_df.columns:
        tx_df["amount"] = tx_df["amount"].apply(clean_currency)

    # Rename users id to client_id for relational joining
    users_df = users_df.rename(columns={"id": "client_id"})

    print(f"  • Users table: {len(users_df)} rows")
    print(f"  • Cards table: {len(cards_df)} rows")
    print(f"  • Transactions table: {len(tx_df)} rows")

    # Step 0: Multi-Table Aggregation
    print("\n--- [Data Integration] Multi-Table Relational Feature Engineering ---")
    card_agg = MultiTableAggregator(
        secondary_df=cards_df,
        join_key="client_id",
        numeric_aggs=["mean", "sum", "count"],
        categorical_aggs=["nunique"],
        prefix="card_",
    )
    card_agg.fit(users_df)
    users_enriched = card_agg.transform(users_df)

    tx_agg = MultiTableAggregator(
        secondary_df=tx_df,
        join_key="client_id",
        numeric_aggs=["mean", "sum", "std", "count", "max"],
        categorical_aggs=["nunique"],
        time_col="date",
        window_days=[30, 90],
        prefix="tx_",
    )
    tx_agg.fit(users_enriched)
    full_dataset = tx_agg.transform(users_enriched)

    print(
        f"✅ Enriched primary table from {len(users_df.columns)} to {len(full_dataset.columns)} features!"
    )

    # Define Target: Predict credit_score
    target_col = "credit_score"
    full_dataset = full_dataset.dropna(subset=[target_col])
    X = full_dataset.drop(columns=[target_col, "address", "client_id"], errors="ignore")
    y = full_dataset[target_col]

    cfg = TaskConfig(
        task_type="regression",
        target=target_col,
        cv_folds=3,
        baseline_gate=True,
        quality_gate=True,
    )

    # Step 1: QC & EDA
    print("\n--- [Step 1] QC & EDA Inspection ---")
    qc = run_qc(X, y=y, cfg=cfg)
    print(qc.report_.summary())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Steps 2-5: Pipeline Training
    print("\n--- [Steps 2-5] Fitting MLKit Pipeline with Model Zoo Benchmark ---")
    pipeline = Pipeline(
        [
            ("type_caster", TypeCaster(drop_id=False)),
            ("imputer", MissingImputer(add_indicator=False)),
            ("encoder", CategoricalEncoder()),
            ("scaler", FeatureScaler()),
            ("var_filter", VarianceFilter(threshold=0.0)),
            ("corr_filter", CorrelationFilter(threshold=0.98)),
            ("arena", ModelArena(cv=3, random_state=42)),
        ]
    )

    pipeline.fit(X_train, y_train, cfg=cfg)
    arena_step = pipeline.named_steps["arena"]
    print("\n🏆 Model Arena Comparison Table:")
    print(arena_step.comparison_table_.to_string(index=False))

    y_pred = pipeline.predict(X_test)
    from sklearn.metrics import mean_squared_error, r2_score

    test_rmse = (mean_squared_error(y_test, y_pred)) ** 0.5
    test_r2 = r2_score(y_test, y_pred)
    print(f"\n📊 Test Set Evaluation: RMSE = {test_rmse:.2f}, R² = {test_r2:.4f}")

    # Step 6: Monitoring & Bundle
    print("\n--- [Step 6] Drift Monitoring & HTML Report Generation ---")
    monitor = DriftMonitor()
    monitor.fit(X_train)
    drift_rep = monitor.monitor(X_test)

    # Generate Diagnostic Visualizations
    fig_arena = plot_arena_comparison(arena_step)
    fig_imp = plot_feature_importance(arena_step, top_n=15)
    fig_corr = plot_correlation_heatmap(X_train.select_dtypes(include=["number"]), top_n=15)

    report_path = Path("reports/banking_pipeline_report.html")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generate_html_report(
        reports=[qc.report_, pipeline.report(), drift_rep],
        title="Banking Multi-Table Credit Score Prediction Dashboard",
        summary_stats={
            "Dataset Size": f"{len(full_dataset)} users",
            "Best Model": arena_step.best_model_name_,
            "Test R²": f"{test_r2:.4f}",
            "Test RMSE": f"{test_rmse:.2f}",
        },
        figures=[
            ("Model Arena Comparison", fig_arena),
            ("Top Feature Importances", fig_imp),
            ("Feature Correlation Heatmap", fig_corr),
        ],
        output_path=report_path,
    )
    print(f"✅ Generated Standalone HTML Report at: {report_path.resolve()}")

    bundle_path = Path("artifacts/banking_bundle")
    save_bundle(pipeline, bundle_path, cfg=cfg, X_sample=X_test.head(10))
    print(f"✅ Saved Production Bundle to: {bundle_path.resolve()}")
    print("\n🎉 Banking Multi-Table Pipeline executed with 100% SUCCESS!")


if __name__ == "__main__":
    main()
