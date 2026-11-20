# ADR-005: mltk là Library Collection, không phải Framework

## Trạng thái
Đã chốt

## Bối cảnh
Người dùng chính là Data Scientist. Cần loại bỏ rào cản học tập phức tạp, không ép buộc cấu trúc dự án cứng nhắc.

## Quyết định
- Sản phẩm là một bộ thư viện độc lập (library collection).
- Data Scientist import các class cụ thể (`MissingImputer`, `LGBMTrainer`, `run_qc`, v.v.) và sử dụng trực tiếp trong script/Jupyter notebook.
- `BaseFrameTransformer` là internal implementation detail để duy trì tính nhất quán và kiểm thử tự động, không phải là thứ DS bắt buộc phải kế thừa.
- YAML cấu hình chỉ là tính năng tùy chọn (optional) cho automation/CI, không phải workflow bắt buộc.

## Hệ quả
- DS có thể tận dụng ngay lập tức mà không cần học cú pháp hay quy tắc trừu tượng.
- Dễ dàng tích hợp với mã nguồn Python tự do của Data Scientist.
