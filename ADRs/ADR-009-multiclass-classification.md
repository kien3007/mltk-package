# ADR-009: Hỗ trợ phân loại nhiều lớp (thay thế ADR-004)

## Trạng thái
Đã chốt (v0.3). Thay thế ADR-004.

## Bối cảnh
ADR-004 giới hạn phiên bản đầu ở bài toán nhị phân và hồi quy. Sau đó, chia theo nhóm/thời gian, hiệu chuẩn và giám
sát đều đã hoàn thiện. Các thư viện cùng loại (scikit-learn, PyCaret, AutoGluon, FLAML) đều hỗ trợ nhiều lớp, và hai
ca benchmark (wine, OpenML cmc) chỉ "đạt" vì bị từ chối đúng cách. Đây là khoảng trống lớn nhất trong bảng năng lực
(`docs/ROADMAP.md`).

## Quyết định
- `prepare_target` mã hóa nhãn nhiều lớp thành 0..K-1 theo thứ tự đã sắp xếp (nhãn số sắp theo giá trị số) và lưu
  `label_map` trong bundle. Giới hạn `modelling.max_classes` (mặc định 50) để chặn trường hợp vô tình dùng target liên
  tục.
- Metric nhiều lớp:
  - precision / recall / F1 dạng macro;
  - ROC-AUC one-vs-rest macro (vẫn là metric chính, nên cách chọn mô hình không đổi);
  - log-loss; Brier nhiều lớp; ECE theo nhãn được dự đoán.
- Xác suất được chuẩn hóa lại theo dòng trước khi tính metric. Bài toán nhị phân có thêm `pr_auc` và `log_loss`.
- Ngưỡng quyết định chỉ áp dụng cho bài toán nhị phân. Với nhiều lớp, dự đoán là lớp có xác suất cao nhất; nếu người
  dùng cấu hình ngưỡng, Arena phát cảnh báo.
- Đầu ra `predict` cho nhiều lớp:
  - `score_<nhãn>` cho từng lớp;
  - `score` = độ tự tin của lớp được chọn;
  - `prediction` (mã lớp) và `label` (nhãn gốc).
- Drift của dự đoán dùng độ tự tin của lớp được chọn.
- `handle_imbalance` cho nhiều lớp dùng `class_weight="balanced"`.

## Hệ quả
- wine (AUC 1,00) và cmc (AUC 0,73) chuyển từ "bị từ chối" sang được huấn luyện; benchmark giữ nguyên số ca đạt.
- Target encoding cho nhiều lớp vẫn dùng frequency encoding; target encoding theo từng lớp được lên kế hoạch ở v0.4.
- Bootstrap CI cho nhiều lớp đi qua đường tính của sklearn (chậm hơn đường numpy dành cho nhị phân).
