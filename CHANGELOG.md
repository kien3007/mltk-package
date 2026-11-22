# Changelog

## [Unreleased] - độ bền tầng dữ liệu (ADR-008)

### Sửa lỗi (phát hiện qua dò lỗi trường hợp biên và dữ liệu thật)
- **Crash:**
  - datetime có timezone và chuỗi ngày trộn nhiều timezone;
  - tên cột trùng (kể cả `1` và `"1"`);
  - histogram số nguyên rất lớn trong dải hẹp; bảng không có feature;
  - cột chứa list/dict trong QC;
  - tên cột one-hot chứa ký tự JSON làm LightGBM crash (SpeedDating).
- **Hiểu sai ý nghĩa:**
  - mã có số 0 đứng đầu bị biến thành số; số điện thoại bị biến thành số;
  - timedelta bị coi là category;
  - ngày lưu dạng số (`yyyymmdd`, epoch, Excel) bị coi là số thường;
  - EDA chưa bỏ qua cột bị đánh dấu `ignore`.

### Tính năng
- `docs/HUONG_DAN_SU_DUNG.md`: tra cứu hàm/class theo từng công đoạn (16 phần, mỗi phần có bảng và ví dụ). Mọi ví dụ
  được chạy tự động (`pytest -m docs`, trong CI).
- `apply_coercion`: một nguồn sự thật cho việc chuyển kiểu, dùng chung bởi profile, TypeCaster, EDA và QC.
- Không crash ở mức từng cột: kiểu `unsupported` / `ignored`; xử lý lỗi theo từng cột, từng cặp, từng panel.
- `TaskConfig.column_types`, `ProfileOptions(overrides=...)`, `mlkit eda --type col=kiểu` (kể cả các dạng
  `datetime:yyyymmdd|epoch_s|epoch_ms|excel|<strftime>`).
- QC mới:
  - `ambiguous_types`, `unsupported_columns`, `duplicate_names`;
  - `leakage_suspect`: rò rỉ một phần qua giá trị (phân loại ≥ 0,97, hồi quy ≥ 0,9) và qua việc giá trị bị thiếu
    (≥ 0,8). Hiệu chỉnh trên dữ liệu thật; không có báo nhầm trên 30 dataset.
- Fuzz test bằng `hypothesis` (CI và `MLKIT_FUZZ_EXAMPLES`); 25 trường hợp biên trở thành test hồi quy.
- `mlkit benchmark --suite openml`: 15 dataset công khai; workflow CI hằng tuần (`benchmark-openml.yml`).
- LightGBM/XGBoost nhận tên cột theo vị trí (`f0..fN`), nên tên cột lạ không còn làm thư viện crash.

## [Unreleased] - nền tảng dữ liệu (ADR-007)

### Hiểu dữ liệu
- `mlkit.qc_eda.profile`: profile và **kiểu ngữ nghĩa** dùng chung cho mọi bước. Nhận diện số lưu dạng chuỗi
  (tiền tệ, %, dấu phân cách, dấu phẩy thập phân), Yes/No, ngày dạng chuỗi, giá trị thiếu đội lốt, `inf`, cột trùng,
  cột ID (theo tên, tách snake/camelCase, không còn nhầm `amount_paid`) và văn bản tự do.
- **QC viết lại trên nền profile**, giữ tương thích tên kiểm tra và tham số cũ.
  - Kiểm tra mới: cột trùng, giá trị thiếu đội lốt, gần như hằng, cột ID, văn bản, kiểu trộn, `inf`, ngoại lai,
    lớp hiếm, mất cân bằng, target hồi quy lệch.
  - `issues_` ghi theo từng cột kèm gợi ý xử lý.
  - **Kiểm tra rò rỉ áp dụng cho mọi cặp kiểu cột và target**; trước đây chỉ cột số với target số.
- `mlkit.qc_eda.eda.run_eda`: EDA cho target nhị phân, nhiều lớp, hồi quy hoặc không có target.
  - Thước đo liên hệ chung trên [0, 1]: |Spearman|, η, Cramér's V có hiệu chỉnh chệch.
  - Bảng target theo bin/category, giá trị thiếu so với target, ma trận liên hệ giữa các cột.
  - Báo cáo HTML độc lập, mọi biểu đồ kèm bảng số liệu.
- `mlkit.viz.eda_charts`: 6 biểu đồ chọn dạng theo kiểu cột, dùng palette đã kiểm chứng và không có trục kép.
- CLI `mlkit eda` và `mlkit benchmark`.

### Tiền xử lý
- `TypeCaster` dựa trên profile:
  - chuyển đổi kiểu và đóng băng kế hoạch chuyển đổi cho dữ liệu mới;
  - loại cột ID, rỗng, hằng, trùng lặp, ghi lý do vào `drop_reasons_`;
  - `time_col` dạng ngày được giữ để tách thành phần lịch.
- `TextFeatures` (mới), `MissingImputer(add_indicator="auto")`, `RareCategoryGrouper(min_count, max_cardinality)`.
- `CategoricalEncoder`:
  - category vừa và nhiều giá trị dùng target encoding; không có target thì dùng frequency encoding (trước đây là
    ordinal với thứ tự tùy ý);
  - `strategies` cho phép chỉ định theo từng cột; ordinal sắp xếp theo thứ tự xác định.
- Pipeline mặc định: `qc → type_caster → datetime → text → imputer → rare_grouper → encoder → ...`.
- **Sửa lỗi:**
  - `RareCategoryGrouper(min_freq=1%)` mặc định đã xóa sạch tín hiệu của cột 400 giá trị (AUC 0,54 lên 0,78);
  - target nhiều lớp dạng số gây crash, giờ báo lỗi rõ ràng;
  - `coerce_numeric` trả về `Float64`/`pd.NA`, giờ là `float64`/`NaN`.

### Hàng rào và hiệu năng
- `mlkit.benchmarks`: 16 dataset, mỗi dataset một kỳ vọng; có job CI riêng. Phiên bản trước đạt 11/16, bản này 16/16.
- QC không chạy lại trong fold CV (`fold_passthrough`); đo độ thuần bằng numpy; TypeCaster chuyển kiểu cột số theo
  khối. Dataset 500 cột: 31 giây xuống 10 giây.

## [0.2.0] - 2026-11-22

### Sửa lỗi tính đúng đắn
- **Rò rỉ target encoding**: `CategoricalEncoder` giờ tính encoding *out-of-fold* (K-fold) cho các dòng train khi gọi `fit_transform`/`Pipeline.fit`. Với target ngẫu nhiên, AUC CV giảm từ ~0.74 (rò rỉ) về ~0.5.
- Target nhiều lớp không còn bị target-encode sai: tự chuyển sang ordinal và kèm cảnh báo (ADR-004).
- `mlkit monitor` trước đây fit monitor trên chính batch mới, nên luôn báo "ổn định". Giờ nó so với baseline đã lưu trong bundle.
- Bundle chỉ lưu baseline của cột category. Giờ lưu đầy đủ: bin PSI, mẫu tham chiếu cho KS, tỷ lệ category và phân phối prediction.
- CV theo thời gian giờ sắp xếp dữ liệu theo `time_col`. Holdout cũng chia theo thời gian, không còn chia ngẫu nhiên.
- `group_col`: tách train/test theo nhóm (`GroupShuffleSplit`), CV dùng `GroupKFold`, và cột nhóm bị loại khỏi đặc trưng.
- Pipeline mặc định chạy `DatetimeFeatures` trước khi ép kiểu/encode. `DatetimeFeatures` nhận diện được cả cột kiểu category/string và `time_col`.
- `TypeCaster` không còn biến NaN thành chuỗi `"nan"`. `MissingImputer` không còn crash khi batch chỉ chứa category mới.
- Các step không còn tích lũy state hoặc report khi `fit` lại.
- Chi-square cho drift category dùng kiểm định hai mẫu (homogeneity) và gộp các ô nhỏ, giảm báo động giả.
- Version ghi trong bundle được lấy từ metadata của package, không còn viết cứng.
- **Tham số `**kwargs` của trainer bị mất khi `clone()`.** CV và Optuna đánh giá một mô hình khác với mô hình được fit
  cuối cùng: 5/9 tham số trong `LGBM_SPACE` và 2/5 trong `RF_SPACE` không hề có tác dụng khi tuning. Các trainer giờ
  dùng chung `BaseTrainer`, trong đó `get_params`/`set_params` có bao gồm `kwargs`.
- `OptunaSearch` refit mô hình thắng bằng *toàn bộ* dict tham số của trial (kể cả hằng số), không chỉ các giá trị được suggest.
  `LGBM_SPACE` thêm `subsample_freq=1` để `subsample` thực sự có tác dụng.

### Củng cố (lượt 3)
- **CV ở mức toàn pipeline là mặc định (ADR-006).** Các bước trước mô hình được fit lại trong từng fold. Với target
  ngẫu nhiên kèm bộ chọn đặc trưng, AUC CV giảm từ 0,693 (rò rỉ) về 0,491. Dùng `Pipeline(cv_scope="model")` để giữ hành vi cũ.
  `CVEvaluator` được tách thành `split` / `evaluate_folds` / `evaluate`.
- Mọi step tuân thủ hợp đồng `clone()` của sklearn. `RatioFeatures`, `GroupAggFeatures` và `MultiTableAggregator`
  trước đây sửa tham số trong `__init__`. `GroupAggFeatures` cũng không reset bảng tra khi fit lại.
- Trainer khai báo kiểu estimator cho sklearn (`__sklearn_tags__` và `_estimator_type`). Trước đây
  `CalibratedClassifierCV` coi trainer của mlkit là regressor.
- Arena: `calibration` (sigmoid/isotonic/auto, kiểm chứng trên cùng các fold, chỉ giữ khi Brier CV giảm) và
  `threshold` (số, `f1` hoặc `youden`, tối ưu trên OOF). Ngưỡng được lưu trong bundle và dùng làm mặc định cho
  `predict`/`serve`; `Arena.predict` áp dụng ngưỡng này.
- `mlkit evaluate`: metric trên dữ liệu có nhãn, tổng thể và theo nhóm (`--by`), so với holdout lúc train và kiểm tra
  `metric_thresholds`. Thoát mã 2 nếu có nhóm không đạt. Batch chỉ có một lớp cho `roc_auc` = NaN (không còn là 0,5).
- `mlkit.features.temporal` (`TemporalSpec`, `build_features`, `build_panel`, `build_labels`): đặc trưng theo mốc
  thời gian dùng chung cho train và chấm điểm. Ví dụ dunnhumby đã chuyển sang dùng module này và cho kết quả trùng
  khớp với bảng viết tay.
- `n_jobs=-1` cho `RandomForestTrainer` và `ModelImportanceSelector`. Thời gian train dunnhumby giảm từ 3 phút 34 giây
  xuống 1 phút 27 giây, kết quả không đổi.

### Tính năng mới
- CLI: `mlkit predict`, `mlkit serve`, `mlkit steps`; `monitor --out/--fail-on`; `run --html-report`; các cờ `-v/-q/--log-json`; mã thoát `2` khi có drift.
- Pipeline khai báo bằng YAML (`steps:`) thông qua registry.
- `metric_thresholds` hoạt động như cổng phát hành trên tập holdout. Hỗ trợ `brier` và `ece` (cận trên), cùng
  `roc_auc`, `accuracy`, `precision`, `recall`, `f1`, `r2` (cận dưới) và `rmse`, `mae` (cận trên). Tên metric không
  hợp lệ bị từ chối trước khi train; lý do không đạt được ghi vào `metrics.json` (`threshold_failures`).
- Metric mới `brier` và `ece` (`expected_calibration_error`) trong `compute_metrics`, CV, khoảng tin cậy bootstrap và holdout.
- `ModelArena(handle_imbalance=...)` / `modelling.handle_imbalance`: tắt class weight khi cần xác suất đã hiệu chuẩn.
- `TypeCaster(drop_time_col=True)`: `time_col` không còn là đặc trưng thô.
- `DriftMonitor(exclude=[...])` / `monitoring.exclude`: bỏ các đặc trưng tăng theo thời gian (tích lũy) khỏi giám sát.
- `metrics.json` chứa kết quả CV của champion, metric holdout, khoảng tin cậy bootstrap và bảng Arena.
- `ModelArena(candidates=[...])`, có hỗ trợ XGBoost (tùy chọn).
- Drift: prediction drift, tỷ lệ category chưa từng thấy, kiểm tra cột bị thiếu; `DriftMonitor.to_dict()/from_dict()`.
- Bundle: `manifest.json` chứa SHA256 và được kiểm tra khi load (ADR-003), kèm `report.json`.
- Module `mlkit.runner` (`train`, `predict_bundle`, `monitor_bundle`), `mlkit.serving` (FastAPI) và `mlkit.tracking` (MLflow).
- Đọc/ghi CSV, Parquet, Feather, Excel, JSON.
- Structured JSON logging với `run_id` lưu trong ContextVar (ADR-002).
- `Pipeline.fit(..., <step>__<param>=...)` route fit params theo kiểu sklearn.

### Đóng gói và hạ tầng
- Dependencies lõi được rút gọn. `xgboost`, `viz`, `explain`, `tuning`, `tracking`, `serve`, `io` chuyển thành extras. `pytest`, `ruff`, `mypy` chuyển sang nhóm `dev`.
- CI có thêm các job lint + mypy, test kèm cổng coverage 80%, cài tối thiểu, chạy ví dụ, build wheel và chạy trên Windows.
- Mypy không còn lỗi nào.
- Version chỉ khai báo một nơi (`src/mlkit/__init__.py`, qua hatch). sdist chỉ gồm mã nguồn, test, ví dụ và tài liệu.
- Áp dụng `ruff format` cho toàn repo; CI kiểm tra `ruff format --check`. `.gitattributes` cố định LF.

### Thay đổi không tương thích
- `ModelArena(models=...)`: thuộc tính `custom_models` vẫn còn để tương thích ngược, nhưng chỉ là alias.
- Định dạng `baseline.json` thay đổi (`schema_version: 2`). Bundle tạo bằng 0.1 cần train lại để dùng được `mlkit monitor`.
- Phân loại nhiều lớp với nhãn dạng chuỗi bị từ chối rõ ràng trong `mlkit run` (ADR-004).
