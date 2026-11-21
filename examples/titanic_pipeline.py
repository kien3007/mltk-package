"""Complete Titanic Model Pipeline using MLTK."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from mltk_features import RatioFeatures
from mltk_modelling import CVEvaluator, LGBMTrainer, load_artifact, save_artifact
from mltk_monitoring import DataDriftMonitor
from mltk_preprocessing import (
    CategoricalEncoder,
    MissingImputer,
    OutlierClipper,
    RareCategoryGrouper,
    TypeCaster,
)
from mltk_qc import read_csv, run_qc
from mltk_selection import (
    CorrelationFilter,
    MissingRateFilter,
    ModelImportanceSelector,
    VarianceFilter,
)


def run_titanic_pipeline() -> None:
    csv_path = r"C:\Users\kienn\Downloads\Titanic-Dataset.csv"
    print(f"=== Bước 1: Đọc dữ liệu Titanic qua mltk-qc: {csv_path} ===")
    df = read_csv(csv_path)
    print(f"Kích thước dataset: {df.shape[0]} dòng, {df.shape[1]} cột")
    print(f"Danh sách cột: {list(df.columns)}")

    # Bước 2: Kiểm tra chất lượng dữ liệu & EDA tự động
    print("\n=== Bước 2: Thực hiện Quality Control (QC & EDA) ===")
    qc_report = run_qc(df, target="Survived")
    os.makedirs("reports", exist_ok=True)
    with open("reports/titanic_qc.html", "w", encoding="utf-8") as f:
        f.write(qc_report.to_html(title="Titanic Dataset QC Report"))
    print("Đã xuất báo cáo HTML trực quan tại: reports/titanic_qc.html")
    print(f"Tổng số kiểm tra: {len(qc_report.results)}")
    for r in qc_report.results:
        status_icon = "✅" if r.passed else ("❌" if r.level == "ERROR" else "⚠️")
        print(f"  {status_icon} [{r.name}]: {r.message}")

    # Bước 3: Tách X và y, phân chia Train/Test
    print("\n=== Bước 3: Phân chia Train / Test (80/20 có Stratify) ===")
    # Loại bỏ PassengerId (ID định danh)
    X = df.drop(columns=["Survived", "PassengerId"])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    # Bước 4: Lắp ráp sklearn Pipeline hoàn chỉnh với các thành phần của MLTK
    print("\n=== Bước 4: Lắp ráp Pipeline biến đổi và huấn luyện ===")
    pipe = Pipeline(
        [
            ("type_caster", TypeCaster()),
            # Tự động loại bỏ cột thiếu > 70% dữ liệu (như Cabin với ~77% missing)
            ("missing_filter", MissingRateFilter(threshold=0.70)),
            # Tự động điền giá trị thiếu (median cho Age/Fare, mode cho Embarked)
            (
                "imputer",
                MissingImputer(numeric_strategy="median", categorical_strategy="most_frequent"),
            ),
            # Cắt giá trị ngoại lai cho Fare
            ("outlier_clipper", OutlierClipper(lower_quantile=0.00, upper_quantile=0.99)),
            # Gom các nhóm hiếm trong các cột phân loại
            ("rare_grouper", RareCategoryGrouper(threshold=0.02, rare_label="__rare__")),
            # Tự động mã hóa: one-hot cho Sex, Embarked; Target/Ordinal cho các cột phức tạp
            (
                "encoder",
                CategoricalEncoder(
                    low_card_threshold=5,
                    high_card_threshold=30,
                    low_strategy="onehot",
                    high_strategy="target",
                    cv=5,
                    random_state=42,
                ),
            ),
            # Lọc bỏ cột hằng hoặc phương sai bằng 0
            ("variance_filter", VarianceFilter(threshold=0.0)),
            # Lọc bỏ cột đa cộng tuyến / tương quan quá cao
            ("corr_filter", CorrelationFilter(threshold=0.95)),
            # Chọn lọc các đặc trưng quan trọng nhất bằng Tree Importance
            ("selector", ModelImportanceSelector(top_k=8, random_state=42)),
            # Huấn luyện LightGBM (tự động điều chỉnh trọng số lớp)
            (
                "model",
                LGBMTrainer(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=4,
                    num_leaves=15,
                    handle_imbalance=True,
                    random_state=42,
                ),
            ),
        ]
    )

    # Bước 5: Đánh giá Cross-Validation trên tập Train
    print("\n=== Bước 5: Đánh giá 5-Fold Stratified Cross-Validation ===")
    evaluator = CVEvaluator(cv=5, task_type="classification", random_state=42)
    cv_summary = evaluator.evaluate(pipe, X_train, y_train)

    print(f"CV ROC-AUC:  {cv_summary['roc_auc_mean']:.4f} ± {cv_summary['roc_auc_std']:.4f}")
    print(f"CV Accuracy: {cv_summary['accuracy_mean']:.4f} ± {cv_summary['accuracy_std']:.4f}")
    print(f"CV F1-Score: {cv_summary['f1_mean']:.4f} ± {cv_summary['f1_std']:.4f}")

    # Bước 6: Fit toàn bộ Pipeline trên tập Train và test trên Test set
    print("\n=== Bước 6: Huấn luyện toàn bộ Pipeline và Đánh giá Test Set ===")
    pipe.fit(X_train, y_train)

    test_preds = pipe.predict(X_test)
    test_probs = pipe.predict_proba(X_test)[:, 1]

    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

    test_acc = accuracy_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test ROC-AUC:  {test_auc:.4f}")
    print("\nBáo cáo phân loại chi tiết (Classification Report):")
    print(classification_report(y_test, test_preds, target_names=["Died", "Survived"]))

    # Bước 7: Lưu Artifacts chuẩn chỉnh phục vụ Production
    print("\n=== Bước 7: Đóng gói và lưu trữ Model Artifacts ===")
    artifact_dir = "./artifacts/titanic_v1"
    save_path = save_artifact(
        pipeline=pipe,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        metrics={
            "cv_roc_auc": cv_summary["roc_auc_mean"],
            "cv_accuracy": cv_summary["accuracy_mean"],
            "test_accuracy": test_acc,
            "test_roc_auc": test_auc,
        },
        output_dir=artifact_dir,
        model_name="titanic_survival_model",
        random_state=42,
    )
    print(f"Đã lưu trọn bộ 7 artifact tại: {save_path}")

    # Bước 8: Tải lại Artifact và kiểm chứng tính toán vẹn
    print("\n=== Bước 8: Tải lại Artifact và kiểm chứng dự đoán ===")
    loaded_pipe, metadata = load_artifact(artifact_dir)
    loaded_preds = loaded_pipe.predict(X_test)
    np.testing.assert_array_equal(test_preds, loaded_preds)
    print(f"✅ Dự đoán của model tải từ artifact khớp 100%!")
    print(f"   Git SHA: {metadata.git_sha}")
    print(f"   Data Hash: {metadata.data_hash}")

    # Bước 9: Giám sát Data Drift trên dữ liệu mới
    print("\n=== Bước 9: Khởi tạo Data Drift Monitor từ Artifact ===")
    monitor = DataDriftMonitor.from_artifact(artifact_dir)
    
    # Giả lập dữ liệu mới phục vụ inference
    drift_report = monitor.evaluate(X_test)
    print(f"Kiểm tra Data Drift trên batch kiểm thử: Passed = {drift_report.passed}")
    for res in drift_report.results[:5]:
        print(f"  [{res.name}] {res.message}")

    print("\n🎉 HOÀN TẤT CHU TRÌNH END-TO-END VỚI DATASET TITANIC!")


if __name__ == "__main__":
    run_titanic_pipeline()
