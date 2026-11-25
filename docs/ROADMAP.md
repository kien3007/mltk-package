# Kế hoạch nâng cấp định kỳ và lộ trình mlkit

Tài liệu sống: mỗi quý rà soát lại **bảng năng lực (mục 1)** và **lộ trình (mục 3)**; mỗi bản phát hành cập nhật cột
"mlkit".

## 1. Năng lực so với các thư viện tham chiếu

Đánh giá ở mức *tính năng chính có sẵn*, dựa trên tài liệu công khai của từng thư viện (11/2026). Mục đích là tìm
khoảng trống, không phải xếp hạng.

Ký hiệu: ✅ có sẵn · ◐ có một phần, hoặc phải tự lắp · — không có.

| Năng lực | mlkit 0.3 | scikit-learn | PyCaret | AutoGluon-Tabular | FLAML | ydata-profiling | Evidently |
|---|---|---|---|---|---|---|---|
| Phân loại nhị phân / nhiều lớp, hồi quy | ✅ | ✅ | ✅ | ✅ | ✅ | — | — |
| Dự báo chuỗi thời gian | ◐ (bảng đặc trưng theo thời điểm) | — | ✅ | ✅ | ✅ | — | — |
| Phân cụm / phát hiện bất thường | — | ✅ | ✅ | — | — | — | — |
| Profile và EDA tự động | ✅ | — | ◐ | — | — | ✅ | ◐ |
| Kiểm tra chất lượng dữ liệu và rò rỉ | ✅ | — | ◐ | — | — | ◐ | ◐ |
| Tự nhận diện kiểu dữ liệu "bẩn" | ✅ | — | ◐ | ✅ | — | ✅ | — |
| CV chống rò rỉ (theo nhóm / thời gian, toàn pipeline) | ✅ | ✅ (tự lắp) | ✅ | ✅ | ◐ | — | — |
| Tối ưu siêu tham số tích hợp | ◐ (`OptunaSearch` tách rời) | ✅ | ✅ | ✅ | ✅ | — | — |
| Ensemble (stacking / blending) | — | ✅ | ✅ | ✅ | ◐ | — | — |
| Preset theo ngân sách thời gian | — | — | ◐ | ✅ | ✅ | — | — |
| Giải thích mô hình (SHAP, lý do cho từng dự đoán) | ◐ (biểu đồ) | ◐ | ✅ | ◐ | — | — | — |
| Hiệu chuẩn xác suất và ngưỡng quyết định | ✅ | ✅ | ✅ | ◐ | — | — | — |
| Cổng phát hành theo metric | ✅ | — | — | — | — | — | ◐ |
| Giám sát drift dữ liệu và dự đoán | ✅ | — | ◐ | — | — | — | ✅ |
| Đánh giá theo nhóm và theo thời gian khi nhãn về | ✅ | — | — | — | — | — | ✅ |
| Phục vụ mô hình qua REST API | ✅ | — | ✅ | ◐ | — | — | — |
| Đặc trưng văn bản / ảnh | ◐ (thống kê văn bản) | ◐ | ◐ | ✅ | — | — | — |
| Dữ liệu lớn hơn RAM / GPU | — | — | ◐ | ◐ | ◐ | ◐ | ◐ |
| Benchmark tính tổng quát trong CI | ✅ | — | — | ◐ | — | — | — |

**Khoảng trống ưu tiên** (có ở hầu hết thư viện cùng loại):
1. Tối ưu siêu tham số tích hợp và preset theo ngân sách thời gian.
2. Ensemble.
3. Giải thích cho từng dự đoán.
4. Dự báo chuỗi thời gian.
5. Khả năng mở rộng cho dữ liệu lớn.

## 2. Quy trình nâng cấp định kỳ

### 2.1 Nhịp

| Chu kỳ | Việc | Cơ chế |
|---|---|---|
| **Hằng tuần** (tự động) | Cập nhật dependency (Python) | Dependabot mở PR (`.github/dependabot.yml`); CI 9 job phải xanh mới merge |
| **Hằng tuần** (tự động) | Benchmark 15 dataset thật | Workflow `benchmark-openml.yml` (thứ Hai); kết quả lưu thành artifact để so sánh giữa các tuần |
| **Hằng tháng** (tự động) | Cập nhật GitHub Actions | Dependabot |
| **Mỗi 4–6 tuần** | Một bản **minor** (`0.x.0`) với **một chủ đề** của lộ trình (mục 3), kèm sửa lỗi | Quy trình phát hành (2.4) |
| **Khi cần** | Bản **patch** (`0.x.y`) cho lỗi nghiêm trọng hoặc lỗ hổng bảo mật | Quy trình phát hành, bỏ qua chủ đề |
| **Hằng quý** | Rà soát bảng năng lực và lộ trình; thêm dataset mới vào benchmark; gỡ các API đã hết hạn deprecation | Cập nhật tài liệu này |

### 2.2 Phiên bản và tương thích
- **SemVer.** Trước 1.0, bản minor được phép đổi API, nhưng **mọi thay đổi không tương thích phải qua một bản
  deprecation**: phát `DeprecationWarning` ở bản `0.x`, gỡ ở bản `0.(x+1)` hoặc muộn hơn. Thay đổi phải ghi trong
  mục "Thay đổi không tương thích" của `CHANGELOG.md`.
- Bundle mang `format_version`. Mỗi khi đổi định dạng phải có hướng dẫn chuyển đổi trong CHANGELOG.
- Quyết định kiến trúc được ghi thành ADR mới trong `ADRs/`, kể cả khi thay thế ADR cũ (ví dụ ADR-009 thay ADR-004).

### 2.3 Cổng chất lượng cho mỗi bản phát hành (Definition of Done)
- [ ] CI xanh: lint, `ruff format`, mypy, test trên Python 3.10–3.12 và Windows, cài tối thiểu, chạy ví dụ, build.
- [ ] Benchmark tổng hợp (`pytest -m benchmark`) và ví dụ trong tài liệu (`pytest -m docs`) đều đạt.
- [ ] Benchmark OpenML đạt 15/15 và **không ca nào giảm quá 0,01 AUC / R²** so với bản trước (so file kết quả).
- [ ] Code mới có test (coverage tổng ≥ 85%); tính năng mới có ít nhất một ca benchmark chứng minh nó chạy được.
- [ ] `docs/HUONG_DAN_SU_DUNG.md` cập nhật cho mọi API mới; `CHANGELOG.md` có mục cho phiên bản.
- [ ] Cột "mlkit" trong bảng năng lực (mục 1) được cập nhật.

### 2.4 Các bước phát hành
1. Tăng `__version__` trong `src/mlkit/__init__.py`; đổi mục "Unreleased" trong CHANGELOG thành `[x.y.z] - ngày`.
2. Merge PR vào `main` (CI xanh).
3. Gắn tag `vx.y.z` và push tag. Workflow `release.yml` sẽ kiểm tra tag khớp `__version__`, chạy test cùng
   benchmark và ví dụ trong tài liệu, build wheel/sdist, rồi tạo GitHub Release kèm file build.
4. (Từ 1.0) Đăng lên PyPI.

## 3. Lộ trình theo mốc

Mỗi mốc có tiêu chí nghiệm thu đo được. Thứ tự có thể đổi sau mỗi lần rà soát quý.

| Mốc | Chủ đề | Nội dung | Nghiệm thu |
|---|---|---|---|
| **v0.3** ✅ | Nhiều lớp và mô hình phổ biến | Phân loại nhiều lớp end-to-end; `HistGBTrainer`, `ExtraTreesTrainer`; metric `pr_auc`, `log_loss`; hạ tầng nâng cấp (Dependabot, workflow phát hành, tài liệu này) | wine và cmc chuyển từ "bị từ chối" sang được huấn luyện (AUC 1,00 / 0,73) |
| **v0.4** | Tối ưu và giải thích | Tối ưu siêu tham số ngay trong Arena (`modelling.tuning: {n_trials, timeout}`); giải thích cho từng dự đoán (top lý do theo SHAP) trong `predict` và `/predict`; permutation importance; target encoding theo từng lớp cho bài toán nhiều lớp | Trung bình AUC/R² trên bộ OpenML tăng; `predict --explain` có test |
| **v0.5** | Ensemble và preset | Stacking/blending top-k trong Arena; preset `fast` / `balanced` / `best_quality` với ngân sách thời gian | Ensemble không kém mô hình đơn tốt nhất trên ≥ 13/15 ca OpenML; preset tôn trọng ngân sách ±10% |
| **v0.6** | Chuỗi thời gian | Bài toán dự báo (`task_type: forecasting`): đặc trưng lag/rolling, backtest theo cửa sổ thời gian; EDA mùa vụ (ACF, phân rã) | Thêm ≥ 3 dataset chuỗi thời gian vào benchmark; vượt baseline "seasonal naive" |
| **v0.7** | Dữ liệu lớn | Backend Polars/PyArrow cho profile và dựng đặc trưng; chấm điểm theo khối; lấy mẫu phân tầng cho EDA | Profile 10 triệu dòng × 50 cột dưới 60 giây và trong 8 GB RAM (ca benchmark mới) |
| **v0.8** | Phi cấu trúc và nhiều bảng | Đặc trưng văn bản bằng embedding (extra `nlp`); tự join bảng quan hệ theo đúng thời điểm | Benchmark có ca văn bản vượt `TextFeatures` hiện tại; ca nhiều bảng không rò rỉ |
| **v1.0** | Ổn định | Đóng băng API công khai; trang tài liệu (mkdocs); đăng PyPI; serialize bằng skops (ADR-003); gỡ toàn bộ API đã deprecated | Không thay đổi không tương thích trong 2 bản minor liên tiếp |

## 4. Nhật ký rà soát

| Ngày | Phiên bản | Ghi chú |
|---|---|---|
| 2026-11-25 | 0.3.0 | Lập kế hoạch; hoàn thành mốc v0.3 |
