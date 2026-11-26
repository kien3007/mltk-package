# ADR-003: Serialization ưu tiên skops thay vì pickle

## Trạng thái
Đã thay thế bởi ADR-012 (v1.0): skops không serialize được DataFrame trong pipeline; dùng chữ ký HMAC.

## Bối cảnh
Pickle tiềm ẩn nguy cơ an ninh (arbitrary code execution) khi load artifact từ nguồn bên ngoài.

## Quyết định
Ưu tiên sử dụng `skops` để serialize/deserialize pipeline; fallback sang `joblib` có kèm metadata hash và version check.

## Hệ quả
- Nâng cao tính an toàn khi chia sẻ model artifact.
- Lưu trữ manifest phiên bản thư viện rõ ràng.
