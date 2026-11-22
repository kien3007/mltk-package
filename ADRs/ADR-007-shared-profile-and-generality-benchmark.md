# ADR-007: Lớp profile dùng chung và benchmark tính tổng quát

## Trạng thái
Đã chốt (v0.3)

## Bối cảnh
QC, EDA, trực quan hóa và tiền xử lý là các bước mà mọi dataset đều phải đi qua. Trước đây mỗi bước tự đoán kiểu cột
theo cách riêng: ID được nhận ra theo tên chứa "id", số lưu dạng chuỗi không được chuyển đổi, ngưỡng gộp category hiếm
cố định 1%. Chạy pipeline mặc định trên 12 dataset khác nhau cho thấy các lỗi gãy âm thầm:
- cột 400 thành phố bị gộp hết vào `__rare__` (AUC 0,54);
- giá tiền dạng `"$1,234"` và văn bản tự do bị coi là category (AUC khoảng 0,48);
- target nhiều lớp dạng số gây crash;
- kiểm tra rò rỉ chỉ áp dụng cho cột số với target số.

## Quyết định
1. **Một lớp profile duy nhất** (`mlkit.qc_eda.profile`) suy ra *kiểu ngữ nghĩa* cho từng cột, gồm: empty, constant,
   boolean, binary, numeric, datetime, categorical, high_cardinality, identifier, text. Lớp này cũng nhận diện số lưu
   dạng chuỗi (tiền tệ, %, dấu phân cách hàng nghìn, dấu phẩy thập phân), giá trị thiếu đội lốt (`n/a`, `null`, `-`),
   giá trị vô cực, cột trùng lặp và tên cột kiểu ID (tách từ theo snake/camelCase).
2. **QC, EDA, TypeCaster và TextFeatures đều đọc cùng profile đó.** TypeCaster đóng băng kế hoạch chuyển kiểu lúc
   `fit` để `transform` áp dụng đúng chuyển đổi đó cho dữ liệu mới.
3. **Kiểm tra rò rỉ áp dụng cho mọi cặp kiểu.** Dùng độ tách thứ hạng `2·|AUC − 0,5|` cho cột số với target nhị phân;
   độ thuần chuẩn hóa theo bin cho các trường hợp phân loại còn lại; η² hoặc |Spearman| cho hồi quy. Mỗi bin cần ít
   nhất 5 dòng và dữ liệu cần ít nhất 30 dòng.
4. **Mặc định thích nghi theo dữ liệu, không dùng hằng số phá kết quả.**
   - Gộp category hiếm theo số lượng tuyệt đối và chỉ ở cột ít giá trị.
   - Category vừa và nhiều giá trị dùng target encoding, hoặc frequency encoding khi không có target; ordinal chỉ dùng
     khi được chỉ định.
   - Cờ thiếu dữ liệu ở chế độ "auto".
   - `time_col` dạng ngày được giữ lại để tách thành các phần lịch.
5. **EDA không bị giới hạn bởi phạm vi mô hình hóa.** Hỗ trợ target nhị phân, nhiều lớp, hồi quy và không có target.
   Thước đo liên hệ chung trên thang [0, 1]: Spearman, η, Cramér's V.
6. **Hàng rào benchmark** (`mlkit.benchmarks`, `mlkit benchmark`, `pytest -m benchmark`, job CI riêng): 16 dataset,
   mỗi dataset một kỳ vọng. Một số ca chỉ đạt được khi tầng nền làm đúng việc của nó.

## Hệ quả
- Phiên bản trước chỉ đạt 11/16 ca benchmark; phiên bản này đạt 16/16. Adult và dunnhumby giữ kết quả trong mức
  dao động bình thường.
- Mọi thay đổi về sau phải giữ benchmark xanh, nên tối ưu cho một dataset không thể âm thầm làm hỏng dataset khác.
- QC không chạy lại trong các fold CV (`fold_passthrough`) và tính toán bằng numpy. Dataset 500 cột train mất khoảng
  10 giây thay vì khoảng 31 giây.
- Phân loại nhiều lớp ở bước mô hình hóa vẫn ngoài phạm vi (ADR-004), nhưng giờ được từ chối nhất quán với thông báo rõ.
