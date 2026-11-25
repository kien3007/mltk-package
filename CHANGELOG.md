# Changelog

## [0.8.0] - 2026-11-25

Các mốc v0.5–v0.8 được phát hành cùng nhau (một PR, tag `v0.8.0`); mỗi mốc có mục riêng bên dưới.

### Văn bản ngữ nghĩa
- `TextFeatures(semantic="tfidf")` (mặc định): TF-IDF từ + n-gram ký tự, giảm chiều bằng TruncatedSVD (32 thành
  phần), chỉ cần scikit-learn. `semantic="embedding"` dùng sentence-transformers nếu đã cài (tự lùi về TF-IDF nếu
  chưa), `semantic=None` giữ hành vi cũ.
- Ca benchmark `semantic_text` (tín hiệu nằm ở từ ngữ, không ở độ dài): AUC 0,50 (chỉ thống kê) → 0,85–0,87.

### Nối nhiều bảng đúng thời điểm (ADR-011)
- `point_in_time_features` / `PointInTimeAggregator`: mỗi dòng chỉ nhận sự kiện của chính đối tượng xảy ra **trước**
  mốc `as_of` của dòng; số lượng, tổng, trung bình, độ lệch chuẩn theo toàn bộ quá khứ và từng cửa sổ; số ngày từ sự
  kiện gần nhất/đầu tiên. Vector hóa hoàn toàn; cộng dồn riêng trong từng đối tượng.
- Test bảo đảm thêm/xóa/sửa sự kiện sau mốc không đổi một giá trị đặc trưng nào. Ca benchmark `multi_table_pit`.

## [0.7.0] - 2026-11-25

### Dữ liệu lớn (ADR-011)
- `ProfileOptions.max_rows` (mặc định 1 triệu): vượt ngưỡng, thống kê từng cột lấy từ mẫu; số dòng, tỉ lệ thiếu, số
  dòng trùng (hash) và cột trùng vẫn chính xác. `DatasetProfile.sampled_rows`.
- `predict_file` / `mlkit predict --chunksize N`: chấm file lớn theo khối, ghi dần ra CSV/Parquet. `iter_table`.
- `read_table(engine="pyarrow")` cho CSV lớn.
- EDA lấy mẫu phân tầng theo target (`stratified_sample`), giữ đủ lớp hiếm.
- `mlkit benchmark --suite scale`: profile 10 triệu dòng × 50 cột trong 30 giây, RAM đỉnh 2,6 GB (tiêu chí < 60 giây,
  < 8 GB).

## [0.6.0] - 2026-11-25

### Dự báo chuỗi thời gian (ADR-010)
- `task_type: forecasting` (chuẩn hóa thành hồi quy + mục `forecasting: {horizon, season, lags, windows}`): đặc trưng
  lag, rolling mean/std và seasonal naive, tất cả chỉ dùng target tại `t - horizon` trở về trước (chiến lược direct).
  `season` tự dò bằng ACF; `lags < horizon` bị từ chối.
- Nhiều chuỗi trong một mô hình (`group_col` = mã chuỗi). Với forecasting, CV luôn chia theo thời gian (không dùng
  GroupKFold theo chuỗi).
- Mỗi lần train ghi `holdout_forecast` (MAE, RMSE, sMAPE, MASE), `seasonal_naive` và `beats_seasonal_naive`.
- `predict` cho bundle dự báo nhận lịch sử + các kỳ cần dự báo (target trống) và trả kèm `time_col` / `group_col`.
- `mlkit.timeseries`: `backtest` (cuốn chiếu nhiều cửa sổ), `make_lag_features`, `forecast_metrics`,
  `autocorrelation`, `detect_season`, `seasonal_decompose` (không cần statsmodels).
- Biểu đồ `plot_acf`, `plot_decomposition`, `plot_forecast`.
- QC không báo rò rỉ cho các cột lag do mlkit tạo.
- Benchmark `mlkit benchmark --suite timeseries`: 3 bộ tổng hợp (offline, chạy trong CI) + 2 bộ OpenML (bike sharing
  theo giờ, CO2 Mauna Loa). Cả 5 bộ đều vượt seasonal naive (skill 0,38–0,69).

## [0.5.0] - 2026-11-25

### Ensemble trong Arena
- `ModelArena(ensemble="vote"|"stack", ensemble_top_k=3)`: soft voting hoặc stacking của top-k candidate, được CV trên
  cùng các fold nên chỉ thắng khi thật sự tốt hơn. `metrics.json["ensemble_members"]`.

### Preset và ngân sách thời gian
- `modelling.preset`: `fast`, `balanced`, `best_quality` (khóa người dùng tự đặt luôn được ưu tiên).
- `modelling.time_budget` (giây) tính cho cả `train()`. Ngân sách dựa trên chi phí CV đo được của từng candidate:
  bỏ bớt candidate, rút ngắn tuning, hạ stacking xuống voting hoặc bỏ ensemble, bỏ hiệu chuẩn/chọn ngưỡng khi không
  còn đủ thời gian. `ModelArena.timings_`, `skipped_`. Tuning dừng trước khi một trial có thể vượt hạn chót và dùng
  lại kết quả CV của trial tốt nhất.

### Nghiệm thu (15 ca OpenML)
- Ensemble stack / vote không kém mô hình đơn tốt nhất ở 15/15 ca; trung bình 0,8527 → 0,8573 (stack).
- `best_quality` với `time_budget=60`: 15/15 ca trong ngân sách (tối đa 52 giây, trung bình 37 giây); trung bình
  0,8639 (1 ca giảm 0,007 < 0,01).

## [0.4.0] - 2026-11-25

### Tối ưu siêu tham số trong Arena
- `modelling.tuning: {n_trials, timeout, models}` chạy Optuna (TPE) trên **cùng các fold** với các candidate khác; mô hình
  tối ưu xuất hiện dưới tên `"<model> (tuned)"` trong bảng so sánh và chỉ thắng khi thật sự tốt hơn trên CV.
- Mặc định tối ưu họ mô hình tốt nhất; hỗ trợ linear, random forest, extra trees, hist GB, LightGBM, XGBoost.
- Tham số tìm được lưu ở `ModelArena.tuned_params_` và `metrics.json["tuned_params"]`.
- Nghiệm thu OpenML (15 ca, `n_trials=15`): trung bình 0,8527 → 0,8573; 8 ca tăng, 1 ca giảm 0,0076 (pima), còn lại giữ nguyên.

### Giải thích dự đoán (`mlkit.explain`)
- `explain_frame(pipeline, X, top_k, method)`: top lý do cho từng dòng (`reason_i`, `reason_i_impact`), quy về cột gốc
  (one-hot/target encoding được cộng dồn). Phương pháp: hệ số tuyến tính, SHAP TreeExplainer, hoặc ablation (mọi mô hình).
- `feature_contributions` và `permutation_importance` (ROC-AUC / R²).
- `predict --explain K`, `predict_bundle(..., explain=K)` và trường `explain` của REST `/predict`.

### Tiền xử lý
- `CategoricalEncoder` chiến lược `target_mc`: target encoding theo từng lớp (out-of-fold) cho bài toán 3–20 lớp.

## [0.3.0] - 2026-11-25

Bao gồm mốc v0.3 (dưới đây) cùng toàn bộ phần **nền tảng dữ liệu** (ADR-007) và **độ bền tầng dữ liệu** (ADR-008)
đã merge sau 0.2.0.

### Phân loại nhiều lớp (ADR-009, thay ADR-004)
- `prepare_target` mã hóa nhãn thành 0..K-1 và lưu `label_map`; giới hạn `modelling.max_classes` (mặc định 50).
- Metric nhiều lớp: precision/recall/F1 macro, ROC-AUC one-vs-rest macro, log-loss, Brier nhiều lớp, ECE theo nhãn
  được dự đoán.
- `predict` trả `score_<nhãn>` cho từng lớp; `evaluate` và `monitor` hỗ trợ nhiều lớp; ngưỡng quyết định chỉ áp dụng
  cho bài toán nhị phân (có cảnh báo khi cấu hình cho nhiều lớp).

### Mô hình và metric
- `HistGBTrainer` (candidate `hist_gb`) và `ExtraTreesTrainer` (candidate `extra_trees`). LightGBM dùng
  `class_weight="balanced"` cho nhiều lớp khi `handle_imbalance`.
- Metric nhị phân mới: `pr_auc` (average precision) và `log_loss`; cả hai dùng được trong `metric_thresholds`
  (`log_loss` theo chiều thấp hơn là tốt hơn).

### Notebook kiểm thử toàn bộ chức năng
- `notebooks/mlkit_full_test.ipynb`: 21 phần, 162 kiểm tra, chạy qua mọi công đoạn (dữ liệu, profile, QC, EDA, 20 biểu
  đồ, tiền xử lý, đặc trưng, chọn đặc trưng, 7 trainer cho 3 loại bài toán, đánh giá, pipeline, bundle, dự đoán, giám
  sát, REST API, MLflow, CLI, bước tự viết, độ bền, benchmark). CI chạy lại qua `tests/test_notebook.py` (`pytest -m docs`).
- Lỗi do notebook phát hiện:
  - các trainer thiếu `score()` theo quy ước sklearn, làm `learning_curve` / `cross_val_score` crash;
  - docstring hướng dẫn backend MLflow dạng file, vốn bị MLflow ≥ 3 từ chối (chuyển sang SQLite).

### Quy trình nâng cấp định kỳ
- `docs/ROADMAP.md`: bảng năng lực so với các thư viện tham chiếu, nhịp nâng cấp, chính sách phiên bản/deprecation,
  cổng chất lượng, lộ trình v0.4 → v1.0.
- `.github/dependabot.yml` (dependency hằng tuần, Actions hằng tháng); `.github/workflows/release.yml` (tag
  `vX.Y.Z` → kiểm tra, build, GitHub Release).

### Thay đổi không tương thích
- Target nhiều lớp không còn bị từ chối, mà được huấn luyện.
- Với nhị phân, `compute_metrics` trả thêm khóa `pr_auc`, `log_loss`.

## Độ bền tầng dữ liệu (ADR-008)

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

## Nền tảng dữ liệu (ADR-007)

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
