# ADR-012: Ký bundle bằng HMAC (thay thế ADR-003)

## Trạng thái
Đã chốt (v1.0). Thay thế ADR-003.

## Bối cảnh
ADR-003 định chuyển sang `skops` để tránh rủi ro thực thi mã khi nạp pickle. Khi thử ở v1.0, `skops.io.dump` thất
bại trên pipeline thật của mlkit, vì skops không serialize được `pandas.DataFrame` (lỗi
`BlockValuesRefs.__reduce_cython__`). Mà pipeline lưu nhiều thuộc tính dạng DataFrame:
- bảng so sánh mô hình;
- bảng tổng hợp quan hệ;
- mẫu dữ liệu.

Muốn dùng skops sẽ phải đổi toàn bộ trạng thái của mọi step sang kiểu cơ bản và vẫn phải khai báo danh sách kiểu
được tin cậy. Như vậy là thay đổi lớn và dễ lỗi, trong khi rủi ro thật là **nạp bundle từ nguồn không đáng tin**.

## Quyết định
- Giữ joblib cho `pipeline.joblib` và giữ manifest SHA256 hiện có (phát hiện file bị sửa sau khi lưu).
- Thêm **chữ ký HMAC-SHA256** của manifest:
  - khóa lấy từ biến môi trường `MLKIT_BUNDLE_KEY` hoặc tham số `signing_key`;
  - khi có khóa, `save_bundle` ghi `manifest.json["signature"]`;
  - `load_bundle` / `verify_bundle` với khóa sẽ **từ chối trước khi unpickle** nếu bundle chưa ký, ký bằng khóa khác,
    hoặc có file bị sửa.
- Không có khóa thì hành vi giữ như trước (chỉ kiểm SHA256), để không phá bundle cũ.

## Hệ quả
- Môi trường production đặt cùng một khóa cho bước train và bước serve, nhờ đó chỉ nạp được bundle do chính mình tạo.
- Chưa phải sandbox: người giữ khóa vẫn tạo được bundle độc hại. Đây là mô hình tin cậy giống ký gói phần mềm.
- Có thể xem xét lại skops khi skops hỗ trợ pandas, hoặc khi trạng thái các step được chuẩn hóa sang kiểu cơ bản.
