"""Multi-Table Relational Pipeline for Banking Credit Risk using MLTK.

Datasets:
- users_data.xlsx (2,000 customers) -> Primary Entity
- cards_data.xlsx (6,146 cards)     -> 1-to-N relationship
- transactions_data.xlsx (157k tx)  -> 1-to-N relationship
- mcc_codes.xlsx (109 categories)   -> Dimension lookup
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from mltk_core import Pipeline, split_data
from mltk_modelling import CVEvaluator, LGBMTrainer, load_artifact, save_artifact
from mltk_monitoring import DataDriftMonitor
from mltk_preprocessing import (
    CategoricalEncoder,
    MissingImputer,
    OutlierClipper,
    TypeCaster,
)
from mltk_qc import run_qc
from mltk_selection import (
    CorrelationFilter,
    MissingRateFilter,
    ModelImportanceSelector,
    VarianceFilter,
)
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def clean_currency(series: pd.Series) -> pd.Series:
    """Clean currency strings like '$127613' to numeric floats."""
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .astype(float)
    )


def build_entity_dataset(banking_dir: str | Path) -> pd.DataFrame:
    """Extract, synthesize, and aggregate 4 relational tables into a single customer-level dataset."""
    base = Path(banking_dir)
    print("--- Đọc 4 bảng quan hệ từ Excel ---")
    df_users = pd.read_excel(base / "users_data.xlsx")
    df_cards = pd.read_excel(base / "cards_data.xlsx")
    df_tx = pd.read_excel(base / "transactions_data.xlsx")
    df_mcc = pd.read_excel(base / "mcc_codes.xlsx")

    print(f"  Users: {df_users.shape[0]} dòng")
    print(f"  Cards: {df_cards.shape[0]} dòng")
    print(f"  Transactions: {df_tx.shape[0]} dòng")
    print(f"  MCC Codes: {df_mcc.shape[0]} dòng")

    # 1. Clean Users
    print("\n--- Làm sạch và trích xuất đặc trưng bảng Users ---")
    df_users["yearly_income"] = clean_currency(df_users["yearly_income"])
    df_users["total_debt"] = clean_currency(df_users["total_debt"])
    df_users["per_capita_income"] = clean_currency(df_users["per_capita_income"])
    df_users["years_to_retirement"] = df_users["retirement_age"] - df_users["current_age"]

    # 2. Aggregate Cards per User (client_id)
    print("--- Tổng hợp đặc trưng thẻ (Cards) về từng khách hàng ---")
    df_cards["credit_limit_num"] = clean_currency(df_cards["credit_limit"])
    df_cards["is_credit"] = (df_cards["card_type"].astype(str).str.lower() == "credit").astype(int)
    df_cards["is_debit"] = (df_cards["card_type"].astype(str).str.lower() == "debit").astype(int)
    df_cards["has_chip_flag"] = (df_cards["has_chip"].astype(str).str.upper() == "YES").astype(int)

    cards_summary = df_cards.groupby("client_id").agg(
        total_credit_limit=("credit_limit_num", "sum"),
        avg_credit_limit=("credit_limit_num", "mean"),
        max_credit_limit=("credit_limit_num", "max"),
        num_cards_total=("id", "count"),
        num_credit_cards_active=("is_credit", "sum"),
        num_debit_cards_active=("is_debit", "sum"),
        chip_card_ratio=("has_chip_flag", "mean"),
        card_brand_diversity=("card_brand", "nunique"),
        cards_issued_total=("num_cards_issued", "sum"),
    ).reset_index()

    # 3. Aggregate Transactions per User (client_id)
    print("--- Tổng hợp đặc trưng giao dịch (Transactions) về từng khách hàng ---")
    df_tx["is_error"] = df_tx["errors"].notnull().astype(int)
    df_tx["is_insufficient_balance"] = (df_tx["errors"] == "Insufficient Balance").astype(int)
    df_tx["is_bad_pin"] = (df_tx["errors"] == "Bad PIN").astype(int)
    df_tx["is_chip_use"] = (df_tx["use_chip"] == "Chip Transaction").astype(int)
    df_tx["is_online_use"] = (df_tx["use_chip"] == "Online Transaction").astype(int)

    tx_summary = df_tx.groupby("client_id").agg(
        total_spent=("amount", "sum"),
        avg_transaction_amt=("amount", "mean"),
        max_transaction_amt=("amount", "max"),
        std_transaction_amt=("amount", "std"),
        transaction_count=("id", "count"),
        error_count=("is_error", "sum"),
        insufficient_balance_count=("is_insufficient_balance", "sum"),
        bad_pin_count=("is_bad_pin", "sum"),
        chip_tx_ratio=("is_chip_use", "mean"),
        online_tx_ratio=("is_online_use", "mean"),
        unique_merchants=("merchant_id", "nunique"),
        unique_mcc_count=("mcc", "nunique"),
        unique_cities=("merchant_city", "nunique"),
    ).reset_index()

    tx_summary["error_rate"] = tx_summary["error_count"] / (tx_summary["transaction_count"] + 1e-4)
    tx_summary["insufficient_balance_rate"] = (
        tx_summary["insufficient_balance_count"] / (tx_summary["transaction_count"] + 1e-4)
    )

    # 4. Join all aggregated features onto Primary Entity (Users)
    print("--- Kết hợp (Join) các bảng tổng hợp vào bảng thực thể chính (Users) ---")
    df_entity = df_users.merge(cards_summary, left_on="id", right_on="client_id", how="left")
    df_entity = df_entity.merge(tx_summary, left_on="id", right_on="client_id", how="left")

    # 5. Ratios across tables
    df_entity["debt_to_income_ratio"] = df_entity["total_debt"] / (df_entity["yearly_income"] + 1.0)
    df_entity["debt_to_credit_limit_ratio"] = df_entity["total_debt"] / (df_entity["total_credit_limit"].fillna(0) + 1.0)
    df_entity["spend_to_income_ratio"] = df_entity["total_spent"].fillna(0) / (df_entity["yearly_income"] + 1.0)

    # 6. Define Target: High Credit Risk (Credit Score < 670 - Subprime cutoff)
    # Target definition is based on standard FICO risk threshold
    df_entity["is_high_risk"] = (df_entity["credit_score"] < 670).astype(int)

    # CRITICAL: Drop credit_score, id, address, client_id columns to avoid target leakage & meaningless features!
    drop_cols = [
        "id",
        "credit_score",  # Target leakage prevention!
        "address",       # High cardinality freeform text
        "latitude",
        "longitude",
        "birth_month",   # Redundant with birth_year
    ]
    if "client_id_x" in df_entity.columns:
        drop_cols.extend(["client_id_x", "client_id_y"])
    elif "client_id" in df_entity.columns:
        drop_cols.append("client_id")

    df_entity.drop(columns=[c for c in drop_cols if c in df_entity.columns], inplace=True)
    return df_entity


def run_banking_pipeline() -> None:
    banking_path = r"C:\Users\kienn\Downloads\banking"
    print("=================================================================")
    print("   BÀI TOÁN TỔNG HỢP NHIỀU BẢNG: DỰ ĐOÁN RỦI RO TÍN DỤNG BANKING")
    print("=================================================================")

    # Bước 1: Tổng hợp quan hệ nhiều bảng về bảng phẳng thực thể khách hàng
    df_entity = build_entity_dataset(banking_path)
    print(f"\nBảng thực thể sau khi phẳng hóa: {df_entity.shape[0]} khách hàng, {df_entity.shape[1]} đặc trưng.")
    print(f"Tỷ lệ khách hàng rủi ro cao (is_high_risk=1): {df_entity['is_high_risk'].mean():.1%}")

    # Bước 2: Kiểm tra chất lượng dữ liệu tự động với mltk-qc
    print("\n=== Bước 2: Rà soát chất lượng dữ liệu bằng mltk-qc ===")
    qc_report = run_qc(df_entity, target="is_high_risk")
    os.makedirs("reports", exist_ok=True)
    with open("reports/banking_qc.html", "w", encoding="utf-8") as f:
        f.write(qc_report.to_html(title="Banking Credit Risk QC Report"))
    print("Đã xuất báo cáo chất lượng tại: reports/banking_qc.html")
    for r in qc_report.results:
        icon = "✅" if r.passed else ("❌" if r.level == "ERROR" else "⚠️")
        print(f"  {icon} [{r.name}]: {r.message}")

    # Bước 3: Phân chia Train/Test bằng mltk_core.split_data
    print("\n=== Bước 3: Phân chia Train/Test 80/20 có Stratify ===")
    X_train, X_test, y_train, y_test = split_data(
        df_entity,
        target="is_high_risk",
        test_size=0.20,
        stratify=True,
        random_state=42,
    )
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Bước 4: Lắp ráp Pipeline biến đổi & Huấn luyện chuẩn MLTK
    print("\n=== Bước 4: Lắp ráp mltk_core.Pipeline ===")
    pipe = Pipeline(
        [
            ("type_caster", TypeCaster()),
            ("missing_filter", MissingRateFilter(threshold=0.50)),
            (
                "imputer",
                MissingImputer(numeric_strategy="median", categorical_strategy="most_frequent"),
            ),
            ("outlier_clipper", OutlierClipper(lower_quantile=0.01, upper_quantile=0.99)),
            ("encoder", CategoricalEncoder(low_card_threshold=5, high_card_threshold=20, random_state=42)),
            ("variance_filter", VarianceFilter(threshold=0.0)),
            ("corr_filter", CorrelationFilter(threshold=0.95)),
            ("importance_selector", ModelImportanceSelector(top_k=15, random_state=42)),
            (
                "model",
                LGBMTrainer(
                    n_estimators=150,
                    learning_rate=0.03,
                    max_depth=5,
                    num_leaves=25,
                    handle_imbalance=True,
                    random_state=42,
                ),
            ),
        ]
    )

    # Bước 5: Đánh giá Cross-Validation 5-Fold
    print("\n=== Bước 5: Đánh giá 5-Fold Stratified Cross-Validation ===")
    evaluator = CVEvaluator(cv=5, task_type="classification", random_state=42)
    cv_res = evaluator.evaluate(pipe, X_train, y_train)

    print(f"CV ROC-AUC:  {cv_res['roc_auc_mean']:.4f} ± {cv_res['roc_auc_std']:.4f}")
    print(f"CV Accuracy: {cv_res['accuracy_mean']:.4f} ± {cv_res['accuracy_std']:.4f}")
    print(f"CV F1-Score: {cv_res['f1_mean']:.4f} ± {cv_res['f1_std']:.4f}")

    # Bước 6: Fit toàn bộ Pipeline và Đánh giá trên tập Test độc lập
    print("\n=== Bước 6: Huấn luyện và Đánh giá trên Test Set ===")
    pipe.fit(X_train, y_train)

    test_preds = pipe.predict(X_test)
    test_probs = pipe.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test ROC-AUC:  {test_auc:.4f}")
    print("\nBáo cáo chi tiết (Classification Report):")
    print(classification_report(y_test, test_preds, target_names=["Good Credit (Low Risk)", "High Credit Risk"]))

    # Bước 7: Đóng gói Artifacts Production
    print("\n=== Bước 7: Đóng gói Artifacts Production ===")
    artifact_dir = "./artifacts/banking_credit_risk_v1"
    save_path = save_artifact(
        pipeline=pipe,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        metrics={
            "cv_roc_auc": cv_res["roc_auc_mean"],
            "cv_accuracy": cv_res["accuracy_mean"],
            "test_accuracy": test_acc,
            "test_roc_auc": test_auc,
        },
        output_dir=artifact_dir,
        model_name="banking_credit_risk_model",
        random_state=42,
    )
    print(f"Artifacts lưu tại: {save_path}")

    # Bước 8: Tải lại và kiểm chứng tính toán vẹn
    print("\n=== Bước 8: Kiểm chứng nạp lại Artifact ===")
    loaded_pipe, metadata = load_artifact(artifact_dir)
    loaded_preds = loaded_pipe.predict(X_test)
    np.testing.assert_array_equal(test_preds, loaded_preds)
    print("✅ Dự đoán model tải từ artifact khớp 100%!")
    print(f"   Data Hash: {metadata.data_hash}")
    print(f"   Git SHA: {metadata.git_sha}")

    # Bước 9: Giám sát Data Drift trên batch kiểm thử
    print("\n=== Bước 9: Giám sát Data Drift từ Artifact ===")
    monitor = DataDriftMonitor.from_artifact(artifact_dir)
    drift_report = monitor.evaluate(X_test)
    print(f"Kết quả kiểm tra Drift: Passed = {drift_report.passed}")
    for res in drift_report.results[:5]:
        print(f"  [{res.name}]: {res.message}")

    print("\n🎉 HOÀN THÀNH XUẤT SẮC BÀI TOÁN NHIỀU BẢNG BANKING VỚI MLTK!")


if __name__ == "__main__":
    run_banking_pipeline()
