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

Yêu cầu: Python >= 3.10 (khuyến nghị 3.11 hoặc 3.12 với [uv](https://github.com/astral-sh/uv)).

```bash
pip install -e .                 # chỉ phần lõi (pandas, sklearn, lightgbm, ...)
pip install -e ".[all]"          # toàn bộ tính năng tùy chọn
uv sync --all-extras             # môi trường phát triển (kèm nhóm dev)
```

| Extra | Nội dung |
|---|---|
| `xgboost` | Ứng viên XGBoost trong Model Arena |
| `viz` | Biểu đồ, HTML report (`matplotlib`, `seaborn`, `plotly`) |
| `explain` | SHAP |
| `tuning` | Tối ưu siêu tham số bằng Optuna |
| `tracking` | Ghi log thí nghiệm lên MLflow |
| `serve` | REST API bằng FastAPI + Uvicorn |
| `io` | Đọc/ghi Parquet, Excel |

---

## 2. Cấu trúc thư viện

```text
src/mlkit/
├─ core/            # config, step, pipeline, registry, bundle, report, log (JSON logging + run_id)
├─ qc_eda/          # Kiểm định chất lượng dữ liệu và EDA
├─ preprocessing/   # TypeCaster, Imputer, Encoder (OOF target encoding), Scaler, ...
├─ features/        # Datetime, Ratio, GroupAgg, Binning, MultiTableAggregator
├─ selection/       # Variance, Correlation, MutualInfo, ModelImportance
├─ modelling/       # Zoo, CVEvaluator (Stratified/Group/TimeSeries), Arena, Optuna, Calibration
├─ monitoring/      # DriftMonitor: PSI, KS, Chi-square 2 mẫu, prediction drift
├─ viz/             # Biểu đồ chẩn đoán + HTML dashboard
├─ runner.py        # train / predict / monitor dạng hàm Python
├─ serving.py       # FastAPI app (extra `serve`)
├─ tracking.py      # MLflow (extra `tracking`)
└─ cli.py           # mlkit run | predict | monitor | serve | steps
```

---

## 3. Chạy qua dòng lệnh (CLI)

```bash
# Huấn luyện + đánh giá holdout + lưu bundle + HTML report
mlkit run examples/adult_config.yaml --out ./artifacts/adult --html-report ./artifacts/adult.html

# Dự đoán cho batch mới (csv / parquet / json / xlsx)
mlkit predict ./artifacts/adult new_data.csv --out preds.csv --threshold 0.5

# Giám sát drift so với baseline đã lưu trong bundle
mlkit monitor ./artifacts/adult new_data.csv --out drift.html --fail-on warn

# Phục vụ model qua REST API (cần extra `serve`)
mlkit serve ./artifacts/adult --port 8000

# Liệt kê các step có thể dùng trong YAML
mlkit steps
```

Cờ chung: `-v` (debug), `-q` (chỉ cảnh báo/lỗi), `--log-json` (log JSON mỗi dòng, có `run_id`, theo ADR-002).

Mã thoát: `0` thành công · `1` lỗi dữ liệu/cấu hình hoặc không đạt `metric_thresholds` · `2` phát hiện drift (`monitor`).

### Cấu hình YAML đầy đủ

```yaml
task_type: classification        # classification (nhị phân) | regression
target: class
id_col: null                     # bị loại khỏi đặc trưng
time_col: null                   # nếu có: holdout theo thời gian + TimeSeriesSplit trong CV
group_col: null                  # nếu có: tách train/test theo nhóm + GroupKFold, cột bị loại khỏi đặc trưng
seed: 42
test_size: 0.2
cv_folds: 5
quality_gate: true
baseline_gate: true
data_path: examples/data/adult.csv

metric_thresholds:               # cổng phát hành trên tập holdout (xem bảng bên dưới)
  roc_auc: 0.85                  # >= ngưỡng
  brier: 0.19                    # <= ngưỡng
  ece: 0.02                      # <= ngưỡng
modelling:
  candidates: [linear, random_forest, lightgbm, xgboost]
  positive_label: ">50K"
monitoring:                      # tham số của DriftMonitor
  psi_threshold_warn: 0.1
  psi_threshold_error: 0.2
tracking:
  mlflow: true
  experiment: adult-income

# Tùy chọn: tự định nghĩa pipeline bằng các khóa trong registry (xem `mlkit steps`)
steps:
  - {name: qc, step: qc.quality_check}
  - {name: dates, step: features.datetime}
  - {name: types, step: preprocessing.type_caster}
  - {name: imputer, step: preprocessing.imputer, params: {add_indicator: true}}
  - {name: encoder, step: preprocessing.categorical_encoder}
  - {name: arena, step: modelling.arena, params: {candidates: [linear, lightgbm]}}
```

### Cổng phát hành `metric_thresholds`

Mỗi ngưỡng được so với metric trên tập holdout. Tên metric không hợp lệ sẽ bị từ chối **trước khi train**
(`ValidationError`), nên gõ sai tên không làm cổng bị bỏ qua âm thầm. Khi không đạt, `mlkit run` thoát với mã `1`,
nhưng bundle vẫn được ghi, kèm `thresholds_passed: false` và `threshold_failures` trong `metrics.json`.

| Loại bài toán | Metric | Điều kiện đạt |
|---|---|---|
| classification | `roc_auc`, `accuracy`, `precision`, `recall`, `f1` | giá trị >= ngưỡng |
| classification | `brier`, `ece` (chất lượng xác suất; nhãn nhị phân 0/1) | giá trị <= ngưỡng |
| regression | `r2` | giá trị >= ngưỡng |
| regression | `rmse`, `mae` | giá trị <= ngưỡng |

`ece` là Expected Calibration Error (10 bin đều); 0 nghĩa là xác suất khớp hoàn toàn với tần suất thực.
Nếu cần dùng điểm số như xác suất, đặt `modelling.handle_imbalance: false`: class weight làm lệch xác suất
(ví dụ dunnhumby 7 ngày: ECE 0,042 với class weight so với 0,010 khi tắt).

### Nội dung bundle

`pipeline.joblib`, `config.yaml`, `schema.json`, `baseline.json` (baseline drift dạng JSON), `metrics.json` (CV, holdout và khoảng tin cậy bootstrap), `report.json`, `versions.json`, `manifest.json` (SHA256 của từng file, được kiểm tra khi load).

> ⚠️ `joblib`/pickle có thể thực thi mã tùy ý khi load. Chỉ load bundle từ nguồn tin cậy. Checksum giúp phát hiện file bị hỏng hoặc bị sửa, nhưng không thay thế được việc tin cậy nguồn.

### REST API

| Endpoint | Mô tả |
|---|---|
| `GET /health` | Trạng thái và model champion |
| `GET /metadata` | Config, schema, metrics, versions |
| `POST /predict` | `{"records": [...], "threshold": 0.5}` |
| `POST /monitor` | `{"records": [...]}`: báo cáo drift cho batch |

---

## 4. Sử dụng trong Python Code

```python
from mlkit.core import TaskConfig
from mlkit.runner import train, predict_bundle, monitor_bundle, read_table

cfg = TaskConfig.from_yaml("examples/adult_config.yaml")
result = train(cfg, read_table(cfg.data_path), output_dir="./artifacts/adult")
print(result.metrics["champion"], result.metrics["holdout"])

preds = predict_bundle("./artifacts/adult", read_table("new_data.csv"))
report = monitor_bundle("./artifacts/adult", read_table("new_data.csv"))
print(report.summary())
```

Hoặc tự lắp pipeline:

```python
from mlkit.core import TaskConfig, Pipeline
from mlkit.features import DatetimeFeatures
from mlkit.preprocessing import TypeCaster, MissingImputer, CategoricalEncoder, FeatureScaler
from mlkit.modelling import ModelArena

cfg = TaskConfig(task_type="classification", target="income", group_col="customer_id")
pipeline = Pipeline([
    ("dates", DatetimeFeatures()),
    ("types", TypeCaster()),
    ("imputer", MissingImputer()),
    ("encoder", CategoricalEncoder()),   # target encoding out-of-fold khi fit
    ("scaler", FeatureScaler()),
    ("arena", ModelArena(cv=5, candidates=["linear", "lightgbm"])),
])
# fit params được route theo kiểu sklearn: <step>__<param>
pipeline.fit(X_train, y_train, cfg=cfg, arena__groups=X_train["customer_id"])
```

---

## 5. Phát triển và kiểm thử

```bash
uv sync --all-extras
uv run pytest                       # kèm cổng coverage >= 80%
uv run ruff check src tests examples
uv run mypy src/mlkit
```

Xem [CHANGELOG.md](CHANGELOG.md) để biết các thay đổi theo phiên bản.
