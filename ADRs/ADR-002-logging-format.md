# ADR-002: Structured JSON Logging với ContextVar run_id

## Trạng thái
Đã chốt

## Bối cảnh
Pipeline chạy nhiều bước cần tracing và logging nhất quán, hỗ trợ phân tích và debug mà không làm thay đổi chữ ký hàm public.

## Quyết định
Sử dụng structured JSON logging (1 dòng 1 JSON object). Quản lý `run_id` xuyên suốt qua `contextvars.ContextVar`.

## Hệ quả
- Truy vết chính xác từng lần chạy qua `run_id`.
- Tương thích tốt với hệ thống logging tập trung (Cloud Logging, ELK).
