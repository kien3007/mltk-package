# ADR-011: Dữ liệu lớn, văn bản ngữ nghĩa và nối bảng đúng thời điểm

## Trạng thái
Đã chốt (v0.7, v0.8).

## Bối cảnh
- Dữ liệu lớn:
  - `profile_dataframe` chạy trên toàn bộ dòng cho mọi cột, và được gọi ở cả QC, EDA và `TypeCaster`;
  - `predict` phải nạp cả file vào RAM.
- `TextFeatures` chỉ có thống kê độ dài, nên văn bản mang nghĩa (sắc thái, chủ đề) gần như vô dụng: AUC 0,50 trên ca
  `semantic_text`.
- `MultiTableAggregator` gộp mọi sự kiện của một đối tượng. Khi mỗi dòng có mốc dự đoán riêng, các sự kiện sau mốc lọt
  vào đặc trưng: rò rỉ tương lai.

## Quyết định
### Dữ liệu lớn
- Thêm `ProfileOptions.max_rows` (mặc định 1 triệu). Vượt ngưỡng này, thống kê từng cột lấy từ mẫu ngẫu nhiên,
  nhưng các đại lượng sau vẫn **chính xác** trên toàn bộ dòng:
  - số dòng;
  - tỉ lệ thiếu (vector hóa);
  - số dòng trùng (một hash 64-bit mỗi dòng);
  - cột trùng (kiểm lại trên toàn bộ).

  `DatasetProfile.sampled_rows` ghi lại việc lấy mẫu.
- Không thêm backend Polars. Pandas + lấy mẫu đã đạt tiêu chí 10M × 50; Polars sẽ nhân đôi bề mặt API và phụ thuộc.
  PyArrow vẫn được dùng ở các chỗ sau:
  - `read_table(engine="pyarrow")`;
  - đọc Parquet theo khối;
  - ghi Parquet tăng dần.
- `predict_file` / `mlkit predict --chunksize N` chấm theo khối, RAM chỉ cần cho một khối.
- EDA lấy mẫu phân tầng theo target (`stratified_sample`) để không mất lớp hiếm.

### Văn bản
- `TextFeatures(semantic="tfidf")` là mặc định: TF-IDF (từ và n-gram ký tự) + TruncatedSVD với 32 thành phần.
  Cách này chỉ cần scikit-learn và được fit lại trong từng fold (pipeline-level CV).
- `semantic="embedding"` dùng sentence-transformers nếu đã cài, thiếu thì lùi về TF-IDF kèm cảnh báo. Không đưa vào
  extras của dự án, vì `uv sync --all-extras` trong CI sẽ kéo theo torch.

### Nhiều bảng
- `point_in_time_features` / `PointInTimeAggregator`: mỗi dòng chỉ nhận sự kiện của chính đối tượng đó **trước**
  mốc `as_of` của dòng, với các cửa sổ trượt.
- Cách cài đặt vector hóa: sắp xếp theo (đối tượng, thời gian), cộng dồn **riêng trong từng đối tượng**, tìm biên bằng
  tìm kiếm nhị phân. Nhờ cộng dồn riêng, ngay cả sai số làm tròn cũng không truyền giữa các đối tượng.
- Có test bảo đảm thêm, xóa hay sửa sự kiện sau mốc không làm đổi một giá trị đặc trưng nào.

## Hệ quả
- Benchmark mới:
  - `--suite scale` (profile 10M × 50: thời gian và RAM đỉnh);
  - `semantic_text` và `multi_table_pit` trong bộ tổng hợp (CI).
- `MultiTableAggregator` được giữ lại cho dữ liệu không có mốc thời gian. Tài liệu chỉ sang `PointInTimeAggregator`
  cho mọi trường hợp có thời gian.
