# Hướng dẫn sử dụng mlkit theo từng công đoạn

Tài liệu này trả lời câu hỏi: **ở mỗi công đoạn, dùng hàm/class nào, để làm gì, đưa vào gì và nhận lại gì.**
Mỗi công đoạn có một bảng tra cứu và một ví dụ ngắn chạy được.

> Các ví dụ được chạy tự động trong CI (`tests/test_docs.py`), dùng dữ liệu mẫu `examples/data/adult.csv`; hãy
> chạy chúng từ thư mục gốc của repo. Cài đặt: `pip install -e ".[all]"` (cần extra `viz` cho biểu đồ và báo cáo HTML).

## Sơ đồ công đoạn

| # | Công đoạn | Module | Hàm/class chính | Lệnh CLI |
|---|---|---|---|---|
| 0 | Đọc dữ liệu, cấu hình | `mlkit.runner`, `mlkit.core` | `read_table`, `TaskConfig`, `prepare_target`, `split_data` | |
| 1 | Profile (nhận diện kiểu cột) | `mlkit.qc_eda` | `profile_dataframe`, `infer_column`, `ProfileOptions` | |
| 2 | Kiểm tra chất lượng (QC) | `mlkit.qc_eda` | `run_qc`, `QualityCheckStep` | `mlkit eda` |
| 3 | EDA | `mlkit.qc_eda` | `run_eda`, `EDAReport` | `mlkit eda` |
| 4 | Trực quan hóa | `mlkit.viz` | `plot_*`, `generate_html_report` | `mlkit eda --out`, `mlkit run --html-report` |
| 5 | Tiền xử lý | `mlkit.preprocessing` | `TypeCaster`, `MissingImputer`, `CategoricalEncoder`, ... | |
| 6 | Tạo đặc trưng | `mlkit.features` | `DatetimeFeatures`, `TextFeatures`, `build_panel`, ... | |
| 7 | Chọn đặc trưng | `mlkit.selection` | `VarianceFilter`, `CorrelationFilter`, `ModelImportanceSelector`, ... | |
| 8 | Huấn luyện và chọn mô hình | `mlkit.modelling` | `ModelArena`, các `*Trainer`, `OptunaSearch` | |
| 9 | Đánh giá | `mlkit.modelling` | `compute_metrics`, `CVEvaluator`, `compute_bootstrap_ci`, ... | |
| 10 | Ghép pipeline và chạy trọn gói | `mlkit.core`, `mlkit.runner` | `Pipeline`, `train`, `build_default_pipeline` | `mlkit run` |
| 11 | Đóng gói | `mlkit.core` | `save_bundle`, `load_bundle` | (tự động trong `mlkit run`) |
| 12 | Dự đoán | `mlkit.runner` | `predict_bundle`, `predict_frame` | `mlkit predict` |
| 13 | Giám sát và đánh giá sau triển khai | `mlkit.monitoring`, `mlkit.runner` | `DriftMonitor`, `monitor_bundle`, `evaluate_bundle` | `mlkit monitor`, `mlkit evaluate` |
| 14 | Phục vụ API, ghi log thí nghiệm | `mlkit.serving`, `mlkit.tracking` | `create_app`, `log_to_mlflow` | `mlkit serve` |
| 15 | Mở rộng và kiểm tra | `mlkit.core`, `mlkit.benchmarks` | `BaseStep`, `register`, `run_suite` | `mlkit steps`, `mlkit benchmark` |

**Quy ước chung cho mọi bước xử lý dữ liệu (`BaseStep`):**
- Có `fit(X, y=None, cfg=None)`, `transform(X)` và `fit_transform(...)`; nhận vào và trả ra `pandas.DataFrame`.
- Không sửa dữ liệu đầu vào.
- Kết quả kiểm tra của bước nằm trong `step.report_` (một `Report`).
- Các thuộc tính có hậu tố `_` (ví dụ `drop_reasons_`) chỉ có sau khi `fit`.

---

## 0. Đọc dữ liệu và cấu hình

| Hàm / class | Dùng để | Tham số chính | Trả về |
|---|---|---|---|
| `read_table(path)` | Đọc file theo đuôi: `.csv`, `.parquet`, `.feather`, `.xlsx`, `.json` | đường dẫn | `DataFrame` |
| `write_table(df, path)` | Ghi file theo đuôi (csv / parquet / json) | `df`, đường dẫn | `Path` |
| `TaskConfig(...)` | Toàn bộ cấu hình của bài toán | `task_type` (`classification`/`regression`), `target`, `id_col`, `time_col`, `group_col`, `cv_folds`, `test_size`, `column_types`, `metric_thresholds`, `modelling`, `monitoring`, `qc`, `steps` | `TaskConfig` |
| `TaskConfig.from_yaml(path)` / `cfg.to_yaml(path)` | Đọc / ghi cấu hình YAML | đường dẫn | `TaskConfig` / — |
| `prepare_target(y, cfg)` | Chuẩn hóa target: nhị phân về 0/1; nhiều lớp về 0..K-1 theo thứ tự đã sắp xếp (tối đa `modelling.max_classes`, mặc định 50) | `y` (Series), `cfg` | `(y, info)`; `info["label_map"]` là ánh xạ nhãn, `info["n_classes"]` là số lớp |
| `split_data(X, y, cfg)` | Chia train/test: theo thời gian nếu có `time_col`, theo nhóm nếu có `group_col`, còn lại ngẫu nhiên phân tầng | `X`, `y`, `cfg` | `Split` (`X_train`, `X_test`, `y_train`, `y_test`, `strategy`) |

```python
from mlkit.core import TaskConfig
from mlkit.runner import prepare_target, read_table, split_data

df = read_table("examples/data/adult.csv")
cfg = TaskConfig(target="class", modelling={"positive_label": ">50K"})
y, info = prepare_target(df["class"], cfg)          # ">50K" -> 1, "<=50K" -> 0
split = split_data(df.drop(columns="class"), y, cfg)
print(info["label_map"], split.strategy, len(split.X_train), len(split.X_test))
```

---

## 1. Profile: nhận diện kiểu từng cột

Mọi công đoạn sau (QC, EDA, tiền xử lý) đều dựa trên profile này.

| Hàm / class | Dùng để | Tham số chính | Trả về |
|---|---|---|---|
| `profile_dataframe(df, y=None, task_type=None, options=None)` | Profile cả bảng và (tùy chọn) target | `df`, `y`, `options` | `DatasetProfile` |
| `infer_column(series, name=None, opts=None)` | Profile một cột | Series | `ColumnProfile` |
| `ProfileOptions(...)` | Tham số nhận diện và **chỉ định kiểu thủ công** | `overrides={"cột": "categorical"}`, `high_cardinality=50`, `sample_size=2000`... | `ProfileOptions` |
| `DatasetProfile` | Kết quả cho cả bảng | thuộc tính `columns`, `target`, `n_rows`, `duplicate_columns`; phương thức `to_frame()`, `columns_of("numeric", ...)`, `to_dict()` | |
| `ColumnProfile` | Kết quả cho một cột | `semantic_type`, `coercion`, `missing_rate`, `n_unique`, `numeric` (thống kê), `notes`, `ambiguous` | |
| `apply_coercion(series, semantic_type, coercion)` | Chuyển giá trị thô theo profile. Đây là hàm dùng chung cho mọi bước. | | Series đã chuyển |
| `coerce_numeric` / `coerce_boolean` / `coerce_datetime` | Chuyển riêng từng kiểu: `"$1,234"` thành 1234.0; `Yes` thành 1.0; `20240101` hoặc chuỗi ngày thành `datetime` | Series (và mã hóa ngày) | Series |
| `normalize_missing(series)` | Đổi `"n/a"`, `"null"`, `"-"`... thành NaN | Series | Series |

**Các kiểu ngữ nghĩa:** `numeric`, `categorical`, `high_cardinality`, `binary`, `boolean`, `datetime`, `text`,
`identifier`, `constant`, `empty`, `ignored`, `unsupported`.

**Giá trị cho `overrides` / `column_types`:** `numeric`, `categorical`, `boolean`, `text`, `identifier`, `ignore`,
`datetime`, `datetime:yyyymmdd`, `datetime:epoch_s`, `datetime:epoch_ms`, `datetime:excel`, `datetime:<strftime>`.

```python
from mlkit.qc_eda import ProfileOptions, profile_dataframe
from mlkit.runner import read_table

df = read_table("examples/data/adult.csv")
prof = profile_dataframe(df.drop(columns="class"), y=df["class"],
                         options=ProfileOptions(overrides={"education-num": "categorical"}))
print(prof.to_frame()[["column", "type", "missing_rate", "n_unique"]].head())
print(prof.columns_of("categorical"), prof.target.kind)
```

---

## 2. Kiểm tra chất lượng (QC)

| Hàm / class | Dùng để | Tham số chính | Trả về |
|---|---|---|---|
| `run_qc(df, y=None, cfg=None, **params)` | Chạy QC một lần để xem kết quả | `enforce_gate=False` để không ném lỗi; các tham số của `QualityCheckStep` | `QualityCheckStep` đã fit |
| `QualityCheckStep(...)` | Bước QC trong pipeline: kiểm tra và **chặn** khi có lỗi | `enforce_gate=True`, `max_missing_rate=0.5`, `max_cardinality=100`, `leakage_threshold=0.99`, `leakage_warn_threshold=0.97`, `leakage_warn_threshold_regression=0.9`, `missing_leakage_warn=0.8`, `imbalance_warn=0.05`, `profile_options` | bước (không đổi dữ liệu) |
| `qc.report_` | Danh sách kết quả từng kiểm tra | `items` (mỗi mục có `name`, `status` = `ok`/`warn`/`error`, `message`, `details`); `has_errors`; `summary()`; `to_dict()` | `Report` |
| `qc.issues_frame_` | Vấn đề theo **từng cột** kèm gợi ý xử lý | | `DataFrame` (`column`, `check`, `severity`, `detail`, `suggestion`) |
| `qc.data_quality_score_` | Điểm chất lượng dữ liệu | | số 0–100 |
| `qc.profile_` | Profile đã dùng | | `DatasetProfile` |
| `univariate_signal(x, y, kind)` | Đo mức một cột "tự quyết định" target, dùng cho kiểm tra rò rỉ | `kind="classification"` hoặc `"regression"` | số trong [0, 1] hoặc `None` |

**Các kiểm tra:**

| Nhóm | Kiểm tra |
|---|---|
| Cấu trúc | `data_shape`, `too_few_rows`, `column_names`, `duplicate_names`, `duplicates`, `duplicate_columns` |
| Giá trị | `missing_values`, `empty_columns`, `disguised_missing`, `infinite_values`, `outliers`, `mixed_types`, `types_as_text` |
| Cột | `constant_columns`, `quasi_constant`, `identifier_columns`, `text_columns`, `high_cardinality`, `unsupported_columns`, `ambiguous_types` |
| Target | `target_distribution`, `target_missing`, `rare_classes`, `class_imbalance` |
| Rò rỉ | `data_leakage` (lỗi), `leakage_suspect` (cảnh báo) |
| Tổng hợp | `data_quality_score` |

```python
from mlkit.qc_eda import run_qc
from mlkit.runner import read_table

df = read_table("examples/data/adult.csv")
qc = run_qc(df.drop(columns="class"), y=df["class"], enforce_gate=False)
print(qc.data_quality_score_, qc.report_.has_errors)
print([(i.status, i.name) for i in qc.report_.items if i.status != "ok"])
print(qc.issues_frame_.head())
```

---

## 3. EDA: phân tích khám phá

| Hàm / class | Dùng để | Tham số chính | Trả về |
|---|---|---|---|
| `run_eda(df, target=None, ...)` | Phân tích toàn diện: profile, QC, liên hệ với target, giá trị thiếu, liên hệ giữa các cột | `target` (tên cột, có thể bỏ), `task_type`, `column_types`, `sample_rows=200_000`, `max_features=30`, `max_categories=20`, `n_bins=10`, `with_qc=True` | `EDAReport` |
| `EDAReport.profile` | Profile của dữ liệu | | `DatasetProfile` |
| `EDAReport.qc` | Kết quả QC | | `QualityCheckStep` |
| `EDAReport.target_association` | Mức liên hệ của từng cột với target, đã xếp hạng | | `DataFrame` (`feature`, `type`, `method`, `strength` ∈ [0, 1], `n`) |
| `EDAReport.target_by_feature[col]` | Tỉ lệ hoặc giá trị trung bình của target theo từng bin/category của cột | | `DataFrame` |
| `EDAReport.missingness` | Tỉ lệ thiếu và target khi thiếu / khi có | | `DataFrame` |
| `EDAReport.associations` | Ma trận liên hệ giữa các cột | | `DataFrame` vuông |
| `EDAReport.summary()` | Tóm tắt dạng chữ | | `str` |
| `EDAReport.figures()` | Toàn bộ biểu đồ | | `list[(tiêu đề, Figure)]` |
| `EDAReport.to_html(path)` | Báo cáo HTML độc lập | đường dẫn | chuỗi HTML |
| `association(a, a_kind, b, b_kind)` | Liên hệ giữa hai cột bất kỳ (`kind` là `"num"` hoặc `"cat"`) | | `(độ mạnh, tên thước đo)` |
| `cramers_v(x, y)` / `correlation_ratio(cat, num)` | Cramér's V (category–category) / tỉ số tương quan η (category–số) | | số trong [0, 1] |

```python
import matplotlib.pyplot as plt

from mlkit.qc_eda import cramers_v, run_eda
from mlkit.runner import read_table

df = read_table("examples/data/adult.csv")
report = run_eda(df, target="class")
print(report.summary())
print(report.target_association.head(5))
print(report.target_by_feature["relationship"])
print(cramers_v(df["sex"], df["class"]))
for title, fig in report.figures():
    plt.close(fig)
report.to_html("artifacts/guide/eda.html")
```

---

## 4. Trực quan hóa

Mọi hàm đều trả về `matplotlib.figure.Figure` và không tự hiển thị. Dùng `fig.savefig(...)` để lưu.

**Biểu đồ cho dữ liệu (EDA):**

| Hàm | Vẽ gì | Đầu vào |
|---|---|---|
| `plot_column_overview(profile)` | Số cột theo kiểu và tỉ lệ thiếu từng cột | `DatasetProfile` |
| `plot_distributions(report.analysed, profile)` | Phân phối từng cột, dạng biểu đồ chọn theo kiểu (histogram, top-category, theo thời gian...) | dữ liệu đã chuyển kiểu của `EDAReport` và profile |
| `plot_target_association(report)` | Xếp hạng mức liên hệ với target | `EDAReport` |
| `plot_target_by_feature(report, col)` | Tỉ lệ target theo nhóm (trên) và số dòng mỗi nhóm (dưới) | `EDAReport`, tên cột |
| `plot_missingness(report)` | Tỉ lệ thiếu và chênh lệch target khi thiếu | `EDAReport` |
| `plot_association_matrix(report)` | Heatmap liên hệ giữa các cột | `EDAReport` |
| `plot_missing_rates(df)`, `plot_target_distribution(y)`, `plot_correlation_heatmap(df)`, `plot_feature_distribution_by_target(df, col, target)` | Biểu đồ nhanh trực tiếp từ `DataFrame` | `DataFrame` / Series |

**Biểu đồ cho mô hình:**

| Hàm | Vẽ gì | Đầu vào |
|---|---|---|
| `plot_arena_comparison(arena)` | So sánh điểm các mô hình | `ModelArena` đã fit |
| `plot_feature_importance(arena, feature_names)` | Độ quan trọng của đặc trưng | `ModelArena` |
| `plot_roc_curve(y, p)`, `plot_precision_recall_curve(y, p)` | Đường ROC (AUC), đường PR (AP) | nhãn, xác suất |
| `plot_calibration_curve(y, p)` | Độ hiệu chuẩn của xác suất | nhãn, xác suất |
| `plot_confusion_matrix(y, pred)` | Ma trận nhầm lẫn | nhãn, dự đoán |
| `plot_cumulative_gains(y, p)`, `plot_lift_curve(y, p)` | Gains và lift (cho bài toán marketing/targeting) | nhãn, xác suất |
| `plot_learning_curve(estimator, X, y)` | Đường học (chẩn đoán underfit/overfit) | mô hình, dữ liệu |
| `plot_shap_summary(estimator, X)` | Độ quan trọng theo SHAP (cần extra `explain`) | mô hình, mẫu dữ liệu |
| `plot_drift_psi(report)` | PSI từng cột từ báo cáo drift | `Report` của `DriftMonitor` |

**Báo cáo HTML:** `generate_html_report(reports, title, summary_stats, figures, output_path)` gộp nhiều `Report` và
biểu đồ thành một trang. `Pipeline.to_html_report(...)` làm việc tương tự cho một pipeline.

```python
import matplotlib.pyplot as plt
import numpy as np

from mlkit.qc_eda import run_eda
from mlkit.runner import read_table
from mlkit.viz import plot_column_overview, plot_roc_curve, plot_target_by_feature

df = read_table("examples/data/adult.csv")
report = run_eda(df, target="class", with_qc=False)
plot_column_overview(report.profile).savefig("artifacts/guide/overview.png")
plot_target_by_feature(report, "education").savefig("artifacts/guide/education.png")

rng = np.random.default_rng(0)
y = rng.integers(0, 2, 500)
plot_roc_curve(y, np.clip(y * 0.6 + rng.uniform(size=500) * 0.5, 0, 1)).savefig("artifacts/guide/roc.png")
plt.close("all")
```

---

## 5. Tiền xử lý (`mlkit.preprocessing`)

Thứ tự khuyến nghị: `TypeCaster` → (`DatetimeFeatures`, `TextFeatures`) → `MissingImputer` → `RareCategoryGrouper`
→ `CategoricalEncoder` → `FeatureScaler`.

| Class | Dùng để | Tham số chính | Thuộc tính sau `fit` |
|---|---|---|---|
| `TypeCaster` | Chuyển kiểu theo profile. Số/ngày/Yes-No lưu dạng chữ được chuyển thành kiểu thật; `"n/a"` và `inf` thành NaN. **Loại** cột ID, rỗng, hằng, trùng, `ignore`, `id_col`/`group_col`/`time_col` (dạng số). | `drop_id=True`, `drop_cols=[...]`, `dtypes={"col": "category"}` (ép kiểu), `profile_options` (chỉ định kiểu), `drop_constant`, `drop_duplicate_columns` | `drop_reasons_`, `inferred_dtypes_`, `semantic_types_` |
| `MissingImputer` | Điền giá trị thiếu | `numeric_strategy` (`median`/`mean`/`zero`), `categorical_strategy` (`mode`/`missing_value`), `add_indicator` (`True`/`False`/`"auto"`: tạo cột `<col>_is_missing` cho cột thiếu ≥ `indicator_min_rate`) | `fill_values_`, `indicator_cols_` |
| `RareCategoryGrouper` | Gộp category hiếm thành `__rare__` | `min_freq` (tỉ lệ), `min_count` (số dòng), `max_cardinality` (bỏ qua cột nhiều giá trị hơn) | `frequent_categories_` |
| `CategoricalEncoder` | Mã hóa category: one-hot (≤ `low_card_threshold`), target encoding **out-of-fold** (nhiều hơn, khi có target), frequency (khi không có target), ordinal (khi chỉ định) | `low_card_threshold=10`, `high_card_threshold=50`, `mid_card_strategy="target"`, `strategies={"col": "ordinal"\|"frequency"\|"target"\|"onehot"}`, `smoothing=10` | `encoding_plan_` |
| `FeatureScaler` | Chuẩn hóa cột số | `method` (`standard`/`robust`/`minmax`) | `numeric_cols_` |
| `OutlierClipper` | Cắt ngoại lai theo IQR | `factor=1.5` | `bounds_` |

```python
from mlkit.core import TaskConfig
from mlkit.preprocessing import CategoricalEncoder, FeatureScaler, MissingImputer, RareCategoryGrouper, TypeCaster
from mlkit.runner import prepare_target, read_table

df = read_table("examples/data/adult.csv")
cfg = TaskConfig(target="class")
X, (y, _) = df.drop(columns="class"), prepare_target(df["class"], cfg)

X1 = TypeCaster().fit_transform(X, cfg=cfg)
X2 = MissingImputer(add_indicator="auto", categorical_strategy="missing_value").fit_transform(X1)
X3 = RareCategoryGrouper(min_freq=None, min_count=10, max_cardinality=10).fit_transform(X2)
enc = CategoricalEncoder()
X4 = enc.fit_transform(X3, y)            # truyền y để có target encoding out-of-fold
X5 = FeatureScaler().fit_transform(X4)
print(enc.encoding_plan_, X5.shape)
```

---

## 6. Tạo đặc trưng (`mlkit.features`)

| Hàm / class | Dùng để | Tham số chính |
|---|---|---|
| `DatetimeFeatures` | Tách cột ngày thành `_year`, `_month`, `_day`, `_dayofweek`, `_is_weekend` (thêm `hour` nếu muốn) | `columns` (mặc định tự nhận), `extract_parts`, `drop_original=True` |
| `TextFeatures` | Thay văn bản tự do bằng `_n_chars`, `_n_words`, `_digit_ratio`, `_upper_ratio`, `_punct_ratio`, `_is_missing` | `columns` (mặc định tự nhận cột kiểu `text`) |
| `RatioFeatures` | Tạo `a_div_b` (và `a_sub_b`) cho các cặp cột số | `pairs=[("a", "b")]`, `create_diff=True` |
| `GroupAggFeatures` | Thống kê của cột số theo nhóm (ví dụ thu nhập trung bình theo nghề) | `group_cols`, `target_cols`, `aggregations=["mean", "std"]` |
| `BinningFeatures` | Chia cột số thành bin theo phân vị (`<col>_bin`) | `columns`, `n_bins=5` |
| `MultiTableAggregator` | Tổng hợp bảng phụ quan hệ 1-nhiều theo khóa. **Không dùng** cho bảng nhiều mốc thời gian (sẽ lọt thông tin tương lai); khi đó dùng `build_panel`. | `secondary_df`, `join_key`, `numeric_aggs`, `window_days` |
| `TemporalSpec` | Mô tả cách tổng hợp nhật ký sự kiện theo thời điểm | `entity_col`, `time_col`, `windows=(7, 28, 90)`, `sum_cols`, `nunique_cols`, `eligibility_days` |
| `build_features(events, as_of, spec)` | Đặc trưng **tại một thời điểm**, chỉ dùng sự kiện có `time ≤ as_of`. Dùng khi chấm điểm. | | `DataFrame` theo đối tượng |
| `build_labels(events, as_of, horizon, spec)` | Nhãn 1/0: có sự kiện trong `(as_of, as_of + horizon]` | | Series |
| `build_panel(events, cutoffs, horizon, spec)` | Bảng huấn luyện: mỗi (đối tượng, cutoff) một dòng, có đặc trưng và nhãn | | `DataFrame` có cột `cutoff`, `target` |

```python
import numpy as np
import pandas as pd

from mlkit.features import DatetimeFeatures, RatioFeatures, TemporalSpec, TextFeatures, build_features, build_panel

df = pd.DataFrame({
    "signup": pd.date_range("2024-01-01", periods=6, freq="10D").strftime("%d/%m/%Y"),
    "review": ["great product, fast delivery"] * 3 + ["bad quality, very slow shipping"] * 3,
    "income": [100.0, 200, 300, 400, 500, 600], "debt": [10.0, 50, 30, 80, 20, 60],
})
from mlkit.preprocessing import TypeCaster

X = TypeCaster().fit_transform(df)               # chuỗi ngày -> datetime
X = DatetimeFeatures().fit_transform(X)
X = TextFeatures(columns=["review"]).fit_transform(X)
X = RatioFeatures(pairs=[("debt", "income")]).fit_transform(X)
print(list(X.columns))

rng = np.random.default_rng(0)
cust, day = np.nonzero(rng.uniform(size=(100, 200)) < rng.uniform(0.02, 0.4, 100)[:, None])
events = pd.DataFrame({"customer": cust, "day": day + 1, "amount": rng.gamma(2, 20, len(cust))})
spec = TemporalSpec(entity_col="customer", time_col="day", windows=(7, 28), sum_cols=("amount",))
panel = build_panel(events, cutoffs=range(60, 190, 7), horizon=7, spec=spec)     # để huấn luyện
now = build_features(events, as_of=200, spec=spec)                                 # để chấm điểm
print(panel.shape, now.columns.tolist()[:6])
```

---

## 7. Chọn đặc trưng (`mlkit.selection`)

| Class | Dùng để | Tham số chính | Thuộc tính sau `fit` |
|---|---|---|---|
| `VarianceFilter` | Bỏ cột có phương sai ≤ ngưỡng (cột hằng) | `threshold=0.0` | `dropped_features_` |
| `CorrelationFilter` | Bỏ một trong hai cột số tương quan cao | `threshold=0.95` | `dropped_features_` |
| `MutualInformationSelector` | Giữ `top_k` cột số có mutual information với target cao nhất (cần `y`) | `top_k=20` | `scores_` |
| `ModelImportanceSelector` | Giữ cột quan trọng theo cây quyết định (cần `y`) | `top_k` hoặc `threshold`, `n_jobs=-1` | `importances_`, `selected_features_` |

Khi các bước có dùng `y` (như hai bước cuối) nằm trong `Pipeline`, chúng được fit lại trong từng fold CV, nên điểm
CV không bị thổi phồng.

```python
from mlkit.core import TaskConfig
from mlkit.preprocessing import CategoricalEncoder, MissingImputer, TypeCaster
from mlkit.runner import prepare_target, read_table
from mlkit.selection import CorrelationFilter, ModelImportanceSelector, VarianceFilter

df = read_table("examples/data/adult.csv")
cfg = TaskConfig(target="class")
y, _ = prepare_target(df["class"], cfg)
X = CategoricalEncoder().fit_transform(MissingImputer().fit_transform(TypeCaster().fit_transform(df.drop(columns="class"))), y)

X = VarianceFilter().fit_transform(X)
X = CorrelationFilter(threshold=0.95).fit_transform(X)
sel = ModelImportanceSelector(top_k=10)
X = sel.fit_transform(X, y, cfg=cfg)
print(sel.selected_features_)
```

---

## 8. Huấn luyện và chọn mô hình (`mlkit.modelling`)

| Hàm / class | Dùng để | Tham số chính | Trả về / thuộc tính |
|---|---|---|---|
| `ModelArena` | **So sánh nhiều mô hình bằng CV** với mô hình ngây thơ; chọn mô hình thắng; tùy chọn hiệu chuẩn xác suất và chọn ngưỡng; fit mô hình thắng trên toàn bộ dữ liệu | `cv=5`, `candidates=["linear", "random_forest", "lightgbm", "xgboost", "hist_gb", "extra_trees"]` (mặc định: 3 mô hình đầu), `models={"tên": estimator}` (thêm mô hình riêng), `handle_imbalance=True`, `calibration=None\|"sigmoid"\|"isotonic"\|"auto"`, `threshold=0.5\|"f1"\|"youden"`, `enforce_baseline_gate=True` | `best_model_name_`, `best_estimator_`, `comparison_table_`, `cv_results_`, `decision_threshold_`, `calibration_`; `predict`, `predict_proba` |
| `LinearTrainer` | Logistic regression (phân loại) / Ridge (hồi quy) | `C`, `handle_imbalance`, `task_type` | mô hình kiểu sklearn |
| `RandomForestTrainer` | Random Forest | `n_estimators`, `max_depth`, `handle_imbalance`, `n_jobs` | |
| `LGBMTrainer` | LightGBM | `n_estimators`, `learning_rate`, `num_leaves`, `handle_imbalance`, `**kwargs` (mọi tham số của LightGBM) | |
| `XGBTrainer` | XGBoost (cần extra `xgboost`) | `n_estimators`, `learning_rate`, `max_depth`, `**kwargs` | |
| `HistGBTrainer` | HistGradientBoosting của scikit-learn: boosting nhanh, tự xử lý giá trị thiếu | `max_iter`, `learning_rate`, `max_leaf_nodes`, `handle_imbalance` | |
| `ExtraTreesTrainer` | Extra Trees: thường chính xác ngang Random Forest nhưng train nhanh hơn | `n_estimators`, `max_depth`, `min_samples_leaf`, `handle_imbalance`, `n_jobs` | |
| `DummyTrainer` | Mô hình ngây thơ (đoán theo tỉ lệ / trung bình) dùng làm mốc so sánh | `task_type` | |
| `SklearnModelWrapper(estimator)` | Dùng bất kỳ mô hình sklearn nào như một trainer của mlkit | estimator | |
| `CalibratedModelWrapper(base, method)` | Hiệu chuẩn xác suất của một mô hình | `method="sigmoid"\|"isotonic"`, `cv=3` | |
| `OptunaSearch(estimator_cls, space, n_trials)` | Tối ưu siêu tham số bằng CV (cần extra `tuning`) | `estimator_cls` (ví dụ `LGBMTrainer`), `space` (`LGBM_SPACE`, `RF_SPACE`, `LINEAR_SPACE` hoặc hàm của bạn), `n_trials`, `cv`, `timeout` | `best_params_`, `best_score_`, `best_estimator_`, `trials_dataframe()` |

Mọi trainer có chung API: `fit(X, y)`, `predict(X)`, `predict_proba(X)` (phân loại), `feature_importances_`. Tất cả
tương thích với `clone()` của scikit-learn.

```python
from mlkit.core import TaskConfig
from mlkit.modelling import LGBM_SPACE, LGBMTrainer, ModelArena, OptunaSearch
from mlkit.preprocessing import CategoricalEncoder, MissingImputer, TypeCaster
from mlkit.runner import prepare_target, read_table

df = read_table("examples/data/adult.csv").sample(2000, random_state=0)
cfg = TaskConfig(target="class")
y, _ = prepare_target(df["class"], cfg)
X = CategoricalEncoder().fit_transform(MissingImputer().fit_transform(TypeCaster().fit_transform(df.drop(columns="class"))), y)

arena = ModelArena(cv=3, candidates=["linear", "lightgbm"], handle_imbalance=False, threshold="f1")
arena.fit(X, y, cfg=cfg)
print(arena.comparison_table_)
print(arena.best_model_name_, arena.decision_threshold_)

search = OptunaSearch(estimator_cls=LGBMTrainer, space=LGBM_SPACE, n_trials=3, cv=3)
search.fit(X, y, cfg=cfg)
print(search.best_params_, round(search.best_score_, 4))
```

---

## 9. Đánh giá

| Hàm / class | Dùng để | Tham số chính | Trả về |
|---|---|---|---|
| `compute_metrics(y_true, y_pred, y_prob=None, task_type)` | Tính metric (nhị phân, nhiều lớp, hồi quy) | phân loại: thêm `y_prob` để có AUC/log-loss/Brier/ECE | `dict`. Nhị phân: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`, `log_loss`, `brier`, `ece`. Nhiều lớp: tương tự, với precision/recall/F1 macro, AUC one-vs-rest macro, không có `pr_auc`. Hồi quy: `rmse`, `mae`, `r2` |
| `compute_bootstrap_ci(y_true, y_pred, y_prob, ...)` | Khoảng tin cậy bootstrap cho từng metric | `n_bootstraps=200`, `ci=0.95` | `dict[metric] = {mean, std, ci_lower, ci_upper}` |
| `expected_calibration_error(y, p)` | ECE: xác suất dự báo khớp với tần suất thực đến đâu (0 là hoàn hảo) | `n_bins=10` | số |
| `optimal_threshold(y, p, criterion)` | Ngưỡng quyết định tối ưu | `"f1"` hoặc `"youden"` | số |
| `CVEvaluator(cv, task_type)` | Cross-validation đúng cách: phân tầng, theo nhóm (`groups`) hoặc theo thời gian (`time_index`) | `.split(X, y, groups=, time_index=)`, `.evaluate(estimator, X, y, ...)`, `.evaluate_folds(estimator, folds)` | chỉ số fold / `dict` metric (`*_mean`, `*_std`, `primary_score`) |

```python
import numpy as np
import pandas as pd

from mlkit.modelling import CVEvaluator, LinearTrainer, compute_bootstrap_ci, compute_metrics
from mlkit.modelling.evaluation import optimal_threshold

rng = np.random.default_rng(0)
X = pd.DataFrame({"x1": rng.normal(size=600), "x2": rng.normal(size=600)})
y = (X["x1"] + rng.normal(scale=0.7, size=600) > 0).astype(int).to_numpy()

p = LinearTrainer().fit(X, y).predict_proba(X)
print(compute_metrics(y, (p[:, 1] >= 0.5).astype(int), p))
print(compute_bootstrap_ci(y, (p[:, 1] >= 0.5).astype(int), p, n_bootstraps=50)["roc_auc"])
print(optimal_threshold(y, p[:, 1], "f1"))

groups = np.repeat(np.arange(60), 10)              # 60 nhóm, CV không trộn nhóm
res = CVEvaluator(cv=3).evaluate(LinearTrainer(), X, y, groups=groups)
print(res["roc_auc_mean"], res["roc_auc_std"])

y3 = rng.integers(0, 3, 600)                        # nhiều lớp: nhãn 0..K-1, xác suất K cột
p3 = LinearTrainer().fit(X.assign(t=y3), y3).predict_proba(X.assign(t=y3))
print(compute_metrics(y3, p3.argmax(axis=1), p3))  # AUC one-vs-rest macro, log_loss, ...
```

---

## 10. Ghép pipeline và chạy trọn gói

| Hàm / class | Dùng để | Tham số chính | Trả về |
|---|---|---|---|
| `Pipeline(steps, cv_scope="pipeline")` | Nối các bước thành pipeline. Với `cv_scope="pipeline"`, CV fit lại mọi bước trong từng fold. | `steps=[("tên", bước), ...]` | `fit(X, y, cfg, **{"bước__tham_số": ...})`, `transform`, `predict`, `predict_proba`, `named_steps`, `report()`, `save(...)` |
| `build_default_pipeline(cfg)` | Pipeline mặc định 12 bước (QC → chuyển kiểu → ngày → văn bản → điền thiếu → gộp hiếm → mã hóa → chuẩn hóa → 3 bước chọn đặc trưng → ModelArena) | `cfg` | `Pipeline` |
| `build_pipeline_from_config(cfg)` | Dựng pipeline từ `cfg.steps` (khóa registry, xem `list_steps()`); nếu không có thì dùng pipeline mặc định | `cfg` | `Pipeline` |
| `train(cfg, df, output_dir=None, html_report=None)` | **Làm trọn gói**: kiểm tra cấu hình → chuẩn hóa target → chia holdout → fit pipeline → đánh giá holdout và CI → kiểm tra `metric_thresholds` → lưu baseline drift → (lưu bundle, báo cáo HTML) | `cfg`, `df` (có cột target) | `TrainResult` (`pipeline`, `metrics`, `monitor`, `drift_report`, `bundle_dir`) |

`result.metrics` gồm: `champion`, `cv`, `holdout`, `holdout_ci`, `arena`, `decision_threshold`, `calibration`,
`thresholds_passed`, `threshold_failures`, `split`, `target`.

```python
from mlkit.core import Pipeline, TaskConfig
from mlkit.features import DatetimeFeatures
from mlkit.modelling import ModelArena
from mlkit.preprocessing import CategoricalEncoder, MissingImputer, TypeCaster
from mlkit.runner import build_default_pipeline, prepare_target, read_table, train

df = read_table("examples/data/adult.csv")
cfg = TaskConfig(target="class", cv_folds=3, modelling={"candidates": ["linear", "lightgbm"]},
                 metric_thresholds={"roc_auc": 0.85})

# Cách 1: trọn gói (đây là việc "mlkit run" làm)
result = train(cfg, df, output_dir="artifacts/guide/adult")
print(result.metrics["champion"], result.metrics["holdout"]["roc_auc"], result.metrics["thresholds_passed"])
print([name for name, _ in build_default_pipeline(cfg).steps])

# Cách 2: tự lắp pipeline
y, _ = prepare_target(df["class"], cfg)
pipe = Pipeline([
    ("types", TypeCaster()), ("dates", DatetimeFeatures()),
    ("imputer", MissingImputer(add_indicator="auto")), ("encoder", CategoricalEncoder()),
    ("arena", ModelArena(cv=3, candidates=["linear"])),
])
pipe.fit(df.drop(columns="class"), y, cfg=cfg)
print(pipe.predict_proba(df.drop(columns="class").head(3))[:, 1])
```

---

## 11. Đóng gói (bundle)

| Hàm | Dùng để | Tham số chính | Trả về |
|---|---|---|---|
| `save_bundle(pipeline, path, cfg, X_sample, baseline, metrics, report)` | Lưu pipeline cùng cấu hình, schema, baseline drift, metric, phiên bản thư viện và checksum (`train(..., output_dir=)` gọi hàm này) | | thư mục bundle |
| `load_bundle(path, verify=True)` | Nạp bundle; kiểm tra checksum và cảnh báo khi phiên bản thư viện khác | | `(pipeline, cfg, metadata)`; `metadata` có `schema`, `baseline`, `metrics`, `report`, `versions` |
| `verify_bundle(path)` | Chỉ kiểm tra tính toàn vẹn | | `dict` checksum; ném `ValidationError` nếu file bị sửa |

> ⚠️ Bundle dùng pickle: chỉ nạp bundle từ nguồn tin cậy.

```python
from mlkit.core import load_bundle
from mlkit.core.bundle import verify_bundle

pipeline, cfg, metadata = load_bundle("artifacts/guide/adult")
print(cfg.target, metadata["metrics"]["champion"], list(metadata["schema"])[:5])
print(len(verify_bundle("artifacts/guide/adult")), "file đã kiểm tra checksum")
```

---

## 12. Dự đoán

| Hàm | Dùng để | Tham số chính | Trả về |
|---|---|---|---|
| `predict_bundle(bundle_path, df, threshold=None)` | Dự đoán từ thư mục bundle | `threshold`: mặc định dùng ngưỡng lưu trong bundle (chỉ áp dụng cho nhị phân) | `DataFrame`: `id_col` (nếu có), `score` (xác suất lớp dương; với nhiều lớp là độ tự tin), `score_<nhãn>` cho từng lớp (nhiều lớp), `prediction`, `label` (nhãn gốc) |
| `predict_frame(pipeline, cfg, metadata, df, threshold=None)` | Như trên nhưng dùng bundle đã nạp sẵn (nhanh khi dự đoán nhiều lần) | | `DataFrame` |
| `validate_input(df, schema)` | Kiểm tra đủ cột lúc train; bỏ cột thừa; sắp lại đúng thứ tự | `schema = metadata["schema"]` | `DataFrame`; ném `ValidationError` nếu thiếu cột |
| `model_output(pipeline, X, task_type)` | Đầu ra dạng số của mô hình: xác suất nhãn dương, hoặc giá trị hồi quy | | mảng |

```python
from mlkit.core import load_bundle
from mlkit.runner import predict_bundle, predict_frame, read_table

new = read_table("examples/data/adult.csv").drop(columns="class").head(5)
print(predict_bundle("artifacts/guide/adult", new))

pipeline, cfg, metadata = load_bundle("artifacts/guide/adult")
print(predict_frame(pipeline, cfg, metadata, new, threshold=0.3)[["score", "prediction"]])
```

---

## 13. Giám sát và đánh giá sau triển khai

| Hàm / class | Dùng để | Tham số chính | Trả về |
|---|---|---|---|
| `monitor_bundle(bundle_path, df)` | So drift của batch mới với baseline lúc train: PSI và KS cho cột số, PSI và chi-square cho category, category mới, cột bị thiếu, drift của dự đoán | | `Report` (`is_passed`, `has_warnings`, `items`, `summary()`, `to_html()`) |
| `DriftMonitor(...)` | Tự tạo baseline và kiểm tra drift | `psi_threshold_warn=0.1`, `psi_threshold_error=0.2`, `exclude=[...]` (cột tăng theo thời gian) | `fit(X_ref)`, `fit_predictions(scores)`, `monitor(X_new, y_pred=None)` → `Report`; `to_dict()` / `from_dict()` để lưu và nạp baseline |
| `calculate_psi(expected, actual)` | Population Stability Index của hai mảng số | `num_buckets=10` | số (< 0,1 ổn định; > 0,2 drift mạnh) |
| `evaluate_bundle(bundle_path, df, by=None)` | Hiệu năng khi đã có nhãn thật: tổng thể và theo nhóm; so với holdout lúc train; kiểm tra `metric_thresholds` | `by` (cột nhóm, ví dụ tuần), `threshold` | `EvaluationResult` (`overall`, `by_group`, `failures`, `reference`, `passed`, `to_dict()`) |
| `evaluate_frame(pipeline, cfg, metadata, df, by)` | Như trên, với bundle đã nạp | | `EvaluationResult` |

```python
import numpy as np

from mlkit.monitoring import DriftMonitor, calculate_psi
from mlkit.runner import evaluate_bundle, monitor_bundle, read_table

df = read_table("examples/data/adult.csv")
shifted = df.assign(age=df["age"] + 15)                       # giả lập drift

print(monitor_bundle("artifacts/guide/adult", shifted.drop(columns="class")).summary())
perf = evaluate_bundle("artifacts/guide/adult", df, by="sex")
print(perf.overall["roc_auc"], perf.passed)
print(perf.by_group[["sex", "n", "roc_auc", "ece"]])

mon = DriftMonitor(exclude=["fnlwgt"]).fit(df.drop(columns="class"))
print(mon.monitor(shifted.drop(columns="class")).has_errors, calculate_psi(df["age"].to_numpy(), shifted["age"].to_numpy()))
```

---

## 14. Phục vụ API và ghi log thí nghiệm

| Hàm | Dùng để | Ghi chú |
|---|---|---|
| `create_app(bundle_path)` | Tạo ứng dụng FastAPI: `GET /health`, `GET /metadata`, `POST /predict` (`{"records": [...], "threshold": tùy chọn}`), `POST /monitor` | cần extra `serve`; chạy server bằng `mlkit serve BUNDLE --port 8000` |
| `log_to_mlflow(train_result, run_name=None)` | Ghi tham số, metric và bundle lên MLflow | cần extra `tracking`; hoặc đặt `tracking: {mlflow: true}` trong cấu hình |

```python
from fastapi.testclient import TestClient

from mlkit.runner import read_table
from mlkit.serving import create_app

client = TestClient(create_app("artifacts/guide/adult"))
records = read_table("examples/data/adult.csv").drop(columns="class").head(2).to_dict(orient="records")
print(client.get("/health").json())
print(client.post("/predict", json={"records": records}).json())
```

---

## 15. Mở rộng và kiểm tra

| Hàm / class | Dùng để | Ghi chú |
|---|---|---|
| `BaseStep` | Lớp cơ sở để viết bước của riêng bạn | Viết `_fit(self, X, y=None, cfg=None)` và `_transform(self, X)`; lưu tham số `__init__` nguyên trạng |
| `register("nhóm.tên")` | Đăng ký bước vào registry để dùng trong YAML (`steps:`) | decorator |
| `get_step(name, **params)` / `list_steps()` | Tạo bước từ registry / liệt kê các khóa | cũng có lệnh `mlkit steps` |
| `Report` | Ghi kết quả kiểm tra trong bước của bạn: `self.report_.add(tên, "ok"\|"warn"\|"error", thông điệp)` | |
| `run_suite(names=None, suite="synthetic"\|"openml")` / `run_case(case)` | Benchmark tính tổng quát: kiểm tra thay đổi của bạn không làm hỏng các dạng dữ liệu khác | cũng có lệnh `mlkit benchmark` |

**Các exception** (đều kế thừa `MLKitError`):

| Exception | Khi nào |
|---|---|
| `QualityGateError` | QC có lỗi |
| `BaselineComparisonError` | Không mô hình nào thắng baseline |
| `ValidationError` | Dữ liệu hoặc cấu hình sai (thiếu cột, quá nhiều lớp, nhãn lạ, sai checksum...) |
| `NotFittedError` | Dùng bước khi chưa `fit` |
| `RegistryError` | Khóa không có trong registry |

```python
import numpy as np

from mlkit.benchmarks import run_suite
from mlkit.core import BaseStep, get_step, list_steps, register


@register("features.log_capital")
class LogCapital(BaseStep):
    """Thêm log(1 + x) cho các cột tiền tệ."""

    def __init__(self, columns=None):
        super().__init__()
        self.columns = columns

    def _fit(self, X, y=None, cfg=None):
        self.columns_ = self.columns or [c for c in X.columns if "capital" in str(c)]
        self.report_.add("log_capital", "ok", f"log-transform {self.columns_}")
        return self

    def _transform(self, X):
        out = X.copy()
        for c in self.columns_:
            out[f"log_{c}"] = np.log1p(out[c].clip(lower=0))
        return out


print("features.log_capital" in list_steps())
step = get_step("features.log_capital")             # dùng được trong YAML: {step: features.log_capital}
print(run_suite(["leaky_feature"], with_eda=False)[["case", "passed"]])
```

---

## Tra cứu lệnh CLI

| Lệnh | Công đoạn | Tương đương trong Python |
|---|---|---|
| `mlkit eda DATA --target T --out F.html [--type C=K]` | 1–4 | `run_eda(...).to_html(...)` |
| `mlkit run CONFIG --out DIR [--html-report F]` | 0, 5–11 | `train(cfg, df, output_dir=...)` |
| `mlkit predict BUNDLE DATA [--out F] [--threshold T]` | 12 | `predict_bundle(...)` |
| `mlkit monitor BUNDLE DATA [--out F] [--fail-on warn]` | 13 | `monitor_bundle(...)` |
| `mlkit evaluate BUNDLE DATA [--by COL] [--out F]` | 13 | `evaluate_bundle(...)` |
| `mlkit serve BUNDLE [--port P]` | 14 | `create_app(...)` |
| `mlkit steps` / `mlkit benchmark [--suite openml]` | 15 | `list_steps()` / `run_suite(...)` |

**Mã thoát:**
- `0`: thành công.
- `1`: lỗi đầu vào/cấu hình, hoặc không đạt cổng chặn.
- `2`: có drift (`monitor`) hoặc vi phạm ngưỡng (`evaluate`).

**Lưu ý về cấu hình:** các mục `preprocessing:`, `features:`, `selection:` trong `TaskConfig` hiện **chưa có tác
dụng**. Để đổi tham số các bước, hãy dùng `steps:`, ví dụ
`- {step: preprocessing.imputer, params: {add_indicator: auto}}`.
