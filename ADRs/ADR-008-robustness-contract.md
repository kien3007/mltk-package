# ADR-008: Cam kết độ bền cho tầng dữ liệu

## Trạng thái
Đã chốt

## Bối cảnh
Benchmark 16 dataset (ADR-007) chỉ chứng minh được 16 *dạng* dữ liệu. Thử thêm 25 trường hợp biên thì 5 trường hợp
crash và 4 trường hợp bị hiểu sai ý nghĩa mà không báo gì:
- **Crash:** datetime có timezone, chuỗi ngày trộn nhiều timezone, tên cột trùng, số nguyên rất lớn trong dải hẹp,
  bảng không có feature.
- **Hiểu sai:** mã có số 0 đứng đầu, số điện thoại, timedelta, ngày lưu dạng số.

Chạy trên 15 dataset thật từ OpenML phát hiện thêm hai vấn đề:
- tên cột one-hot chứa ký tự JSON làm LightGBM crash (SpeedDating);
- rò rỉ một phần không bị phát hiện (Titanic `boat`; employee_salaries `2016_gross_pay_received`).

"Cover mọi case" không thể bảo đảm tuyệt đối. Điều có thể cam kết và kiểm chứng được là các mục dưới đây.

## Quyết định
1. **Không crash ở mức từng cột.** Profile không bao giờ ném lỗi: cột không phân tích được có kiểu `unsupported`,
   lý do ghi trong `notes`, và bị loại khỏi feature. EDA, biểu đồ, QC và TypeCaster đều xử lý lỗi theo từng cột,
   từng cặp liên hệ, từng panel.
2. **Một nguồn sự thật cho việc chuyển kiểu.** `apply_coercion` được profile, TypeCaster, EDA và QC dùng chung.
   - Datetime luôn là naive UTC.
   - Ngày lưu dạng số: `yyyymmdd` nhận theo giá trị; epoch và Excel chỉ nhận khi tên cột gợi ý là ngày.
   - Timedelta đổi sang giây.
   - Mã có số 0 đứng đầu được giữ dạng chữ.
   - Khoảng trắng bên trong không bị bỏ, nên số điện thoại không bị biến thành số.
3. **Người dùng có quyền quyết định cuối cùng.** `TaskConfig.column_types`, `ProfileOptions(overrides=...)` và
   `mlkit eda --type col=kiểu` được QC, EDA và tiền xử lý tôn trọng như nhau. Cột nhập nhằng được ghi
   `ambiguous:` và QC cảnh báo `ambiguous_types`.
4. **Fuzz test bằng `hypothesis`.** Ngẫu nhiên hóa tên cột (kể cả trùng, unicode, số), 23 kiểu cột và 7 kiểu
   target. Các thuộc tính phải giữ:
   - profile, QC, EDA và biểu đồ không bao giờ ném lỗi;
   - chuỗi tiền xử lý luôn cho ra dữ liệu sẵn sàng cho mô hình (toàn số, hữu hạn, không NaN);
   - schema ổn định khi gặp batch mới.

   CI chạy 60/20/60 ví dụ; đặt `MLKIT_FUZZ_EXAMPLES=400` để tìm sâu hơn ở máy cục bộ.
5. **Benchmark dữ liệu thật** (`mlkit benchmark --suite openml`, workflow hằng tuần): 15 dataset công khai.
6. **Cảnh báo rò rỉ một phần (`leakage_suspect`).** Lỗi (chặn pipeline) vẫn ở mức tín hiệu ≥ 0,99. Cảnh báo:

   | Loại tín hiệu | Ngưỡng cảnh báo |
   |---|---|
   | Giá trị, phân loại (độ tách thứ hạng) | ≥ 0,97 |
   | Giá trị, hồi quy (\|Spearman\| / η²) | ≥ 0,9 |
   | Việc giá trị bị thiếu | ≥ 0,8 |

   Các ngưỡng được hiệu chỉnh trên dữ liệu thật:
   - đặc trưng hợp lệ mạnh nhất quan sát được: phân loại 0,951 (breast cancer `worst perimeter`), hồi quy 0,81
     (house prices `OverallQual`);
   - rò rỉ đã biết: Titanic `boat` (thiếu: 0,943), employee_salaries `2016_gross_pay_received` (Spearman 0,909).

   Trên 30 dataset của hai bộ benchmark, cảnh báo xuất hiện đúng ở 2 trường hợp rò rỉ đã biết và không báo nhầm.
   Ngưỡng dựa trên số ít dataset, nên có thể chỉnh qua tham số của `QualityCheckStep`.

## Hệ quả
- 25/25 trường hợp biên chạy được và trở thành test hồi quy.
- Benchmark: 16/16 tổng hợp và 15/15 OpenML; fuzz 930 bảng ngẫu nhiên không lỗi.
- Giới hạn còn lại:
  - dữ liệu ngoài dạng bảng (ảnh, âm thanh, JSON lồng nhau);
  - dữ liệu nhiều bảng cần join; dữ liệu vượt quá RAM;
  - EDA chuyên cho chuỗi thời gian; ngữ nghĩa văn bản;
  - category mã hóa bằng số nguyên nhiều giá trị (Amazon) vẫn được coi là số, nếu cần thì người dùng chỉ định
    `column_types`.
