# Changelog

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

### Tính năng mới
- CLI: `mlkit predict`, `mlkit serve`, `mlkit steps`; `monitor --out/--fail-on`; `run --html-report`; các cờ `-v/-q/--log-json`; mã thoát `2` khi có drift.
- Pipeline khai báo bằng YAML (`steps:`) thông qua registry.
- `metric_thresholds` hoạt động như cổng phát hành trên tập holdout. Hỗ trợ `brier` và `ece` (cận trên), cùng
  `roc_auc`, `accuracy`, `precision`, `recall`, `f1`, `r2` (cận dưới) và `rmse`, `mae` (cận trên). Tên metric không
  hợp lệ bị từ chối trước khi train; lý do không đạt được ghi vào `metrics.json` (`threshold_failures`).
- Metric mới `brier` và `ece` (`expected_calibration_error`) trong `compute_metrics`, CV, khoảng tin cậy bootstrap và holdout.
- `ModelArena(handle_imbalance=...)` / `modelling.handle_imbalance`: tắt class weight khi cần xác suất đã hiệu chuẩn.
- `TypeCaster(drop_time_col=True)`: `time_col` không còn là đặc trưng thô.
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

### Thay đổi không tương thích
- `ModelArena(models=...)`: thuộc tính `custom_models` vẫn còn để tương thích ngược, nhưng chỉ là alias.
- Định dạng `baseline.json` thay đổi (`schema_version: 2`). Bundle tạo bằng 0.1 cần train lại để dùng được `mlkit monitor`.
- Phân loại nhiều lớp với nhãn dạng chuỗi bị từ chối rõ ràng trong `mlkit run` (ADR-004).
