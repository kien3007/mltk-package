"""Example demonstrating multi-model training and comparison in MLTK.

Shows how to compare multiple model architectures using MLTK's unified trainer interface:
- LGBMTrainer (LightGBM)
- XGBTrainer (XGBoost)
- RandomForestTrainer (Random Forest with auto-balancing)
- LinearTrainer (Logistic Regression / Ridge)
- SklearnModelWrapper (Generic wrapper for any custom scikit-learn estimator)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from mltk_core import Pipeline, split_data
from mltk_modelling import (
    CVEvaluator,
    LGBMTrainer,
    LinearTrainer,
    RandomForestTrainer,
    SklearnModelWrapper,
    XGBTrainer,
    save_artifact,
)
from mltk_preprocessing import (
    CategoricalEncoder,
    FeatureScaler,
    MissingImputer,
    OutlierClipper,
    TypeCaster,
)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def run_multi_model_pipeline() -> None:
    data_path = Path(r"C:\Users\kienn\Downloads\Titanic-Dataset.csv")
    if not data_path.exists():
        data_path = Path("Titanic-Dataset.csv")

    print(f"Loading data from: {data_path}")
    raw_df = pd.read_csv(data_path)

    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    df = raw_df.drop(columns=[c for c in drop_cols if c in raw_df.columns])

    X_train, X_test, y_train, y_test = split_data(
        df,
        target="Survived",
        test_size=0.2,
        stratify=True,
        random_state=42,
    )

    models: dict[str, Any] = {
        "LightGBM": LGBMTrainer(n_estimators=100, num_leaves=15, random_state=42),
        "XGBoost": XGBTrainer(n_estimators=100, max_depth=4, random_state=42),
        "Random Forest": RandomForestTrainer(n_estimators=100, max_depth=6, random_state=42),
        "Logistic Regression": LinearTrainer(C=0.5, random_state=42),
        "Extra Trees (via Wrapper)": SklearnModelWrapper(
            estimator=ExtraTreesClassifier(n_estimators=100, random_state=42)
        ),
    }

    evaluator = CVEvaluator(cv=5, random_state=42)
    comparison_records = []

    print("\n" + "=" * 65)
    print("      BENCHMARK SO SÁNH ĐA MÔ HÌNH VỚI MLTK-MODELLING")
    print("=" * 65)

    for name, model_trainer in models.items():
        steps = [
            ("type_caster", TypeCaster()),
            ("imputer", MissingImputer()),
            ("outlier", OutlierClipper()),
            ("encoder", CategoricalEncoder()),
        ]
        if "Logistic" in name:
            steps.append(("scaler", FeatureScaler()))
        steps.append(("model", model_trainer))

        pipe = Pipeline(steps)

        # 1. 5-Fold Cross Validation
        cv_res = evaluator.evaluate(pipe, X_train, y_train)

        # 2. Fit on full train set and evaluate on test set
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)
        prob_1 = y_prob[:, 1] if y_prob.ndim == 2 else y_prob

        test_auc = roc_auc_score(y_test, prob_1)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)

        comparison_records.append(
            {
                "Model": name,
                "CV ROC-AUC": f"{cv_res['roc_auc_mean']:.4f} ± {cv_res['roc_auc_std']:.4f}",
                "Test ROC-AUC": round(test_auc, 4),
                "Test Accuracy": f"{test_acc * 100:.2f}%",
                "Test F1": round(test_f1, 4),
            }
        )

    summary_df = pd.DataFrame(comparison_records)
    print("\nBảng so sánh hiệu năng các mô hình:")
    print(summary_df.to_string(index=False))

    best_model_name = summary_df.sort_values(by="Test ROC-AUC", ascending=False).iloc[0]["Model"]
    print(f"\n🏆 Mô hình chiến thắng: {best_model_name}")

    # Đóng gói artifact của mô hình chiến thắng
    best_trainer = models[best_model_name]
    best_pipe = Pipeline(
        [
            ("type_caster", TypeCaster()),
            ("imputer", MissingImputer()),
            ("encoder", CategoricalEncoder()),
            ("model", best_trainer),
        ]
    )
    best_pipe.fit(X_train, y_train)

    art_dir = save_artifact(
        pipeline=best_pipe,
        X_train=X_train,
        y_train=y_train,
        output_dir="./artifacts/best_benchmark_model",
        model_name=f"best_{best_model_name.lower().replace(' ', '_')}",
    )
    print(f"✅ Đã đóng gói artifact mô hình '{best_model_name}' tại: {art_dir}")


if __name__ == "__main__":
    from typing import Any

    run_multi_model_pipeline()
