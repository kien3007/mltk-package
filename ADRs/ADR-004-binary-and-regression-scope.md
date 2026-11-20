# ADR-004: Phạm vi bài toán v1 giới hạn ở phân loại nhị phân và hồi quy

## Trạng thái
Đã chốt

## Bối cảnh
Cần tối ưu hóa phạm vi để đảm bảo chất lượng, tốc độ hoàn thiện và tính ổn định cao nhất cho phiên bản đầu tiên.

## Quyết định
Phiên bản v1 tập trung hoàn chỉnh cho bài toán tabular nhị phân (binary classification) và hồi quy (regression). Các bài toán chuỗi thời gian (time-series split) và group-split được dời sang v2.

## Hệ quả
- Thiết kế CV đơn giản, chắc chắn: StratifiedKFold và KFold.
- Đánh giá metrics rõ ràng, tối ưu cho tabular data.
