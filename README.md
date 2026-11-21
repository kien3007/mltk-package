# MLKit: Thư viện 6 bước xây dựng mô hình Machine Learning dạng bảng

`mlkit` là thư viện Python chuẩn hóa và đóng gói trọn vẹn 6 giai đoạn trong quy trình xây dựng mô hình học máy dạng bảng (Tabular ML):
1. **QC & EDA**: Kiểm định chất lượng dữ liệu, kiểm tra schema, missing, outlier, rò rỉ dữ liệu (leakage), và cổng chất lượng (Quality Gate).
2. **Preprocessing**: Tự động nhận diện kiểu, impute kèm cờ missing, gộp category hiếm, mã hóa theo cardinality (One-Hot, Ordinal, Target Encoding), chuẩn hóa (Scaler).
3. **Feature Engineering**: Tạo đặc trưng thời gian (Datetime), tỷ lệ/hiệu (Ratio/Diff), tổng hợp nhóm (Group Aggregation), phân thùng (Binning).
4. **Feature Selection**: Lọc phương sai (Variance), loại tương quan cao (Collinearity), Mutual Information, độ quan trọng từ mô hình (Model Importance).
5. **Modelling**: Model Zoo (Dummy baseline, Linear, Random Forest, LightGBM, XGBoost, Scikit-learn wrapper), Cross-Validation, so sánh đa mô hình (Model Arena), và cổng so với baseline.
6. **Monitoring**: Thống kê tham chiếu (Baseline Snapshot), tính toán chính xác chỉ số ổn định dân số PSI, kiểm định Kolmogorov-Smirnov (KS-test), và Chi-Square.

---

## 1. Cài đặt nhanh

Yêu cầu: Python >= 3.10 (Khuyến nghị 3.11 hoặc 3.12 với [uv](https://github.com/astral-sh/uv)).

```bash
# Cài đặt thư viện ở chế độ editable
pip install -e .

# Hoặc dùng uv:
uv sync --all-packages
```

---

## 2. Cấu trúc thư viện

```text
mlkit/
├─ pyproject.toml               # Cấu hình dự án, dependencies, CLI entrypoint
├─ src/mlkit/
│  ├─ core/                     # config (TaskConfig), step (BaseStep), report, pipeline, registry, bundle, errors
│  ├─ qc_eda/                   # Kiểm định chất lượng dữ liệu và EDA
│  ├─ preprocessing/            # Imputer, Scaler, Encoder, RareCategoryGrouper, OutlierClipper
│  ├─ features/                 # Datetime, Ratio, GroupAgg, Binning
│  ├─ selection/                # Variance, Correlation, MutualInfo, ModelImportance
│  ├─ modelling/                # Zoo (Dummy, Linear, RF, LightGBM, XGBoost, SklearnWrapper), CVEvaluator, Arena
│  ├─ monitoring/               # DriftMonitor, exact PSI calculation, KS-test, Chi-square
│  └─ cli.py                    # Giao diện dòng lệnh (CLI): mlkit run / mlkit monitor
├─ tests/                       # Unit tests và kiểm thử trường hợp biên
├─ examples/                    # Ví dụ thực thi trên 2 dataset chuẩn
│  ├─ data/                     # Adult Census và California Housing
│  ├─ adult_config.yaml         # Config phân loại (Classification)
│  ├─ adult_classification.py   # Script pipeline phân loại
│  ├─ housing_config.yaml       # Config hồi quy (Regression)
│  └─ california_regression.py # Script pipeline hồi quy
└─ README.md
```

---

## 3. Chạy qua dòng lệnh (CLI)

Thư viện tích hợp sẵn lệnh CLI `mlkit run`:

```bash
# 1. Chạy bài toán phân loại trên Adult Census
mlkit run examples/adult_config.yaml --out ./artifacts/adult_bundle

# 2. Chạy bài toán hồi quy trên California Housing
mlkit run examples/housing_config.yaml --out ./artifacts/housing_bundle

# 3. Giám sát data drift cho một batch dữ liệu mới
mlkit monitor ./artifacts/adult_bundle examples/data/adult.csv
```

---

## 4. Sử dụng trong Python Code

```python
import pandas as pd
from mlkit.core import TaskConfig, Pipeline
from mlkit.qc_eda import QualityCheckStep
from mlkit.preprocessing import TypeCaster, MissingImputer, CategoricalEncoder, FeatureScaler
from mlkit.selection import VarianceFilter, CorrelationFilter
from mlkit.modelling import ModelArena

# 1. Khai báo config
cfg = TaskConfig(task_type="classification", target="income", cv_folds=5, seed=42)

# 2. Lắp ráp pipeline
pipeline = Pipeline([
    ("qc", QualityCheckStep(enforce_gate=True)),
    ("type_caster", TypeCaster()),
    ("imputer", MissingImputer()),
    ("encoder", CategoricalEncoder()),
    ("scaler", FeatureScaler()),
    ("variance", VarianceFilter()),
    ("corr", CorrelationFilter()),
    ("modelling", ModelArena(cv=5, enforce_baseline_gate=True)),
])

# 3. Fit và chọn model tối ưu tự động
pipeline.fit(X_train, y_train, cfg=cfg)

# 4. Dự đoán và xuất bundle
preds = pipeline.predict(X_test)
pipeline.save(path="./artifacts/my_model_bundle", cfg=cfg)
```

---

## 5. Chạy Kiểm thử (Test Suite)

Thư viện bao phủ đầy đủ unit tests cho cả 6 giai đoạn và các trường hợp biên:

```bash
uv run pytest -v
```
