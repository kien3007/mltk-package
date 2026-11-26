# MLKit: Thư viện 6 bước xây dựng mô hình Machine Learning dạng bảng

> 📘 **[Hướng dẫn sử dụng theo từng công đoạn](docs/HUONG_DAN_SU_DUNG.md)**: mỗi công đoạn (profile, QC, EDA,
> trực quan hóa, tiền xử lý, đặc trưng, chọn đặc trưng, mô hình, đánh giá, đóng gói, dự đoán, giám sát) dùng
> hàm/class nào, để làm gì, tham số và kết quả trả về, kèm ví dụ chạy được.
>
> 🧪 **[Notebook kiểm thử toàn bộ chức năng](notebooks/mlkit_full_test.ipynb)**: 21 phần, 162 kiểm tra, có sẵn
> output; CI chạy lại notebook mỗi lần thay đổi.
>
> 🗺️ **[Kế hoạch nâng cấp định kỳ và lộ trình](docs/ROADMAP.md)**: so sánh năng lực với các thư viện cùng loại,
> nhịp phát hành, cổng chất lượng, các mốc v0.4 → v1.0.

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
pip install mltk-package          # chỉ phần lõi (pandas, sklearn, lightgbm, ...)
pip install "mltk-package[all]"   # toàn bộ tính năng tùy chọn
# trong code vẫn là: import mlkit

# phát triển từ mã nguồn
pip install -e ".[all]"
uv sync --all-extras             # môi trường phát triển (kèm nhóm dev)
```

Giấy phép: MIT (xem `LICENSE`). Tài liệu: <https://kien3007.github.io/mltk-package/>.

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

## 3. Hiểu dữ liệu trước khi mô hình hóa: profile, QC, EDA (ADR-007)

Mọi bước (QC, EDA, biểu đồ, tiền xử lý) đọc chung một **profile** gán cho mỗi cột một *kiểu ngữ nghĩa*:
`numeric`, `categorical`, `high_cardinality`, `boolean`, `binary`, `datetime`, `text`, `identifier`, `constant`, `empty`.
Profile tự nhận ra:
- số lưu dạng chuỗi (`"$1,234.50"`, `"12%"`, `"3,25"`);
- `Yes`/`No` và ngày lưu dạng chuỗi;
- giá trị thiếu đội lốt (`"n/a"`, `"null"`, `"-"`) và `inf`;
- cột trùng lặp và cột ID (theo tên lẫn theo độ duy nhất).

```bash
# Báo cáo HTML độc lập: tổng quan, chất lượng dữ liệu, phân phối, phân tích target, giá trị thiếu, liên hệ giữa các cột
mlkit eda data.csv --target churn --out eda.html --profile-json profile.json
```

```python
from mlkit.qc_eda import run_eda, profile_dataframe, run_qc

report = run_eda(df, target="churn")       # target nhị phân / nhiều lớp / hồi quy / không có target
print(report.summary())
report.target_association                    # mức liên hệ [0,1]: |Spearman| / η / Cramér's V tùy cặp kiểu
report.target_by_feature["city"]             # tỉ lệ target theo category (top N + other)
report.to_html("eda.html")

qc = run_qc(df.drop(columns="churn"), y=df["churn"], enforce_gate=False)
qc.issues_frame_                             # cột, kiểm tra, mức độ, chi tiết, gợi ý xử lý
```

**Các kiểm tra QC:**
- kích thước dữ liệu, tên cột, dòng trùng, cột trùng;
- giá trị thiếu, cột rỗng, giá trị thiếu đội lốt;
- cột hằng và gần như hằng, cột ID, văn bản tự do;
- cardinality cao, kiểu lưu dạng chuỗi, kiểu trộn lẫn, `inf`, ngoại lai;
- target: rỗng, hằng, thiếu, lớp quá hiếm so với số fold, mất cân bằng, hồi quy lệch;
- **rò rỉ cho mọi cặp kiểu cột và target**;
- điểm chất lượng dữ liệu (0–100).

**Tiền xử lý mặc định dựa trên profile:**

| Bước | Hành vi |
|---|---|
| `type_caster` | Chuyển đổi các kiểu trên; loại cột ID, rỗng, hằng, trùng lặp, `id_col`, `group_col`, và `time_col` dạng số (lý do loại lưu ở `drop_reasons_`). |
| `datetime` | Tách ngày thành các phần lịch. |
| `text` | Thay văn bản tự do bằng các thống kê: độ dài, số từ, tỉ lệ chữ số, chữ hoa, dấu câu. |
| `imputer` | Tạo cờ thiếu tự động; category thiếu trở thành một giá trị riêng. |
| `rare_grouper` | Chỉ gộp mức hiếm (< 10 dòng) ở cột ít giá trị. |
| `encoder` | One-hot cho cột ≤ 10 giá trị; target encoding OOF cho cột nhiều hơn; frequency khi không có target; ordinal chỉ khi chỉ định. |

### Độ bền và chỉ định kiểu cột (ADR-008)

- **Không crash ở mức từng cột:** cột không phân tích được có kiểu `unsupported`, lý do được ghi lại, và cột bị
  loại khỏi feature thay vì làm hỏng cả bảng.
- **Nhận đúng ý nghĩa** cho:
  - datetime có timezone (đổi về UTC);
  - ngày lưu dạng số (`20240101`; epoch hay Excel khi tên cột gợi ý là ngày);
  - timedelta (đổi sang giây);
  - mã có số 0 đứng đầu (giữ dạng chữ) và số điện thoại (không biến thành số).
- **Cảnh báo `leakage_suspect`** cho rò rỉ một phần: cột chỉ được ghi sau khi biết kết quả, hoặc biến đại diện
  cho target.
- **Khi suy luận không chắc chắn, bạn quyết định kiểu cột**, và mọi bước đều dùng cùng quyết định đó:

```yaml
column_types:                 # trong config YAML (TaskConfig)
  stars: categorical          # số nguyên ít giá trị nhưng thực chất là mã
  order_day: datetime:yyyymmdd
  created: datetime:epoch_s   # cũng có: epoch_ms, excel, hoặc định dạng strftime như datetime:%d/%m/%Y
  notes: ignore
```

```bash
mlkit eda data.csv --target y --type stars=categorical --type order_day=datetime:yyyymmdd
```

### Benchmark tính tổng quát

```bash
mlkit benchmark                 # 18 dataset, mỗi dataset một kỳ vọng (thoát mã 1 nếu có ca trượt)
mlkit benchmark --suite openml  # 15 dataset thật từ openml.org (cần mạng; CI chạy hằng tuần)
mlkit benchmark --suite timeseries  # 5 bài dự báo, phải vượt baseline seasonal naive
mlkit benchmark --suite scale   # profile 10 triệu dòng x 50 cột: thời gian và RAM đỉnh
MLKIT_FUZZ_EXAMPLES=400 uv run pytest tests/test_robustness.py -k fuzz   # fuzz test sâu
uv run pytest -m benchmark      # cùng bộ ca, dạng test (CI chạy ở job riêng)
```

Bộ ca gồm: sklearn (breast cancer, diabetes, wine), mất cân bằng 1%, 400 category, kiểu dữ liệu bẩn, văn bản tự do,
thiếu dữ liệu có ý nghĩa, hiệu ứng theo thứ trong tuần, dữ liệu theo nhóm, dữ liệu rộng (500 cột), hồi quy lệch,
tên cột là số, toàn category, rất nhỏ (80 dòng), văn bản mang nghĩa (sắc thái nằm ở từ ngữ), bảng giao dịch
nối đúng thời điểm, và một feature rò rỉ (QC phải chặn).

## 3b. Chạy qua dòng lệnh (CLI)

```bash
# Huấn luyện + đánh giá holdout + lưu bundle + HTML report
mlkit run examples/adult_config.yaml --out ./artifacts/adult --html-report ./artifacts/adult.html

# Dự đoán cho batch mới (csv / parquet / json / xlsx)
mlkit predict ./artifacts/adult new_data.csv --out preds.csv --threshold 0.5

# Giám sát drift so với baseline đã lưu trong bundle
mlkit monitor ./artifacts/adult new_data.csv --out drift.html --fail-on warn

# Đo hiệu năng trên dữ liệu đã có nhãn, tổng thể và theo từng nhóm (backtest hoặc giám sát khi nhãn về)
mlkit evaluate ./artifacts/adult labelled.csv --by week --out backtest.json

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
  handle_imbalance: false        # true: class weight (lệch xác suất); false: xác suất dùng được trực tiếp
  calibration: auto              # null | sigmoid | isotonic | auto: chỉ giữ khi Brier CV giảm
  threshold: f1                  # 0.5 (mặc định) | số trong (0,1) | f1 | youden (tối ưu trên OOF)
monitoring:                      # tham số của DriftMonitor
  psi_threshold_warn: 0.1
  psi_threshold_error: 0.2
  exclude: [tenure_days]         # bỏ đặc trưng tăng theo thời gian khỏi giám sát drift
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

### Cross-validation ở mức toàn pipeline (ADR-006)

Mặc định, trong mỗi fold CV, mọi bước trước mô hình (encoder, scaler, bộ chọn đặc trưng...) được fit lại chỉ trên
phần train của fold. Nhờ đó điểm CV, việc chọn mô hình, hiệu chuẩn và ngưỡng quyết định đều không bị rò rỉ
nhãn validation. Ví dụ: với target ngẫu nhiên, AUC CV là 0,49 so với 0,69 khi chỉ CV ở mức mô hình.
Dùng `Pipeline(steps, cv_scope="model")` nếu cần tốc độ thay vì độ chính xác của CV.

### Đặc trưng theo mốc thời gian (propensity / churn / repurchase)

```python
from mlkit.features import TemporalSpec, build_panel, build_features

spec = TemporalSpec(entity_col="customer_id", time_col="day", windows=(7, 28, 90),
                    sum_cols=("amount",), nunique_cols=("basket_id",))
panel = build_panel(events, cutoffs=range(120, 700, 7), horizon=7, spec=spec)  # train: nhiều cutoff
X_now = build_features(events, as_of=711, spec=spec)                           # chấm điểm: một mốc
```

Chỉ các sự kiện có `time <= as_of` được đọc (có test bảo đảm). Nhãn là có hay không sự kiện trong `(as_of, as_of + horizon]`.
Cutoff nào có cửa sổ nhãn vượt quá dữ liệu sẽ bị từ chối, để tránh âm tính giả. Train và chấm điểm dùng chung một hàm.
Xem `examples/dunnhumby_propensity.py`.

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
