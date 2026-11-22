# ADR-006: Cross-validation ở mức toàn pipeline là mặc định

## Trạng thái
Đã chốt (v0.2)

## Bối cảnh
`ModelArena` trước đây cross-validate trên dữ liệu **đã được biến đổi** bởi các bước phía trước. Các bước có giám sát
(chọn đặc trưng theo độ quan trọng, target encoding) vì thế được fit trên cả những dòng sau này dùng làm validation.
Kết quả kiểm chứng: với target hoàn toàn ngẫu nhiên và `ModelImportanceSelector(top_k=10)` trên 400 đặc trưng nhiễu,
AUC CV đạt 0,693, trong khi giá trị đúng phải xấp xỉ 0,5. Holdout không bị ảnh hưởng, nhưng điểm CV trong
`metrics.json` và việc chọn mô hình thắng thì có.

## Quyết định
- `Pipeline(cv_scope="pipeline")` là mặc định. Khi bước cuối hỗ trợ (`ModelArena.supports_fold_data`), Pipeline lấy
  định nghĩa fold từ `ModelArena.make_splits` (Stratified, Group hoặc TimeSeries như trước), rồi trong **mỗi fold**
  fit một bản `clone()` của toàn bộ các bước phía trước chỉ trên các dòng train của fold. Arena chấm điểm trên các
  fold này (`evaluate_folds`).
- Sau CV, các bước gốc vẫn được fit một lần trên toàn bộ tập train, như trước.
- `cv_scope="model"` giữ hành vi cũ (nhanh hơn, nhưng điểm CV có thể lạc quan).
- Mọi step phải tuân thủ hợp đồng `clone()` của sklearn: `__init__` lưu tham số nguyên trạng, không sao chép hay gán
  mặc định. Có test kiểm tra toàn bộ registry.

## Hệ quả
- Với target nhiễu nói trên, AUC CV giảm từ 0,693 về 0,491.
- Chi phí: các bước phía trước được fit `cv + 1` lần. Trên dunnhumby (155 nghìn dòng, 4 fold), thời gian tăng từ khoảng
  2 phút 24 giây lên 3 phút 34 giây. Sau khi bật `n_jobs=-1` cho Random Forest và ExtraTrees, thời gian còn khoảng
  1 phút 27 giây.
- Hiệu chuẩn và chọn ngưỡng quyết định của Arena dùng chung các fold này, nên cũng không bị rò rỉ.
- Bổ sung cho ADR-004: chia theo nhóm và theo thời gian, vốn được dời sang v2, đã được hỗ trợ từ v0.2 và dùng chung
  một định nghĩa fold.
