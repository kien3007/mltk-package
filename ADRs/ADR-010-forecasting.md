# ADR-010: Dự báo chuỗi thời gian trên nền pipeline dạng bảng

## Trạng thái
Đã chốt (v0.6).

## Bối cảnh
Dự báo theo thời gian (doanh số, tải điện, nhu cầu) là loại bài toán phổ biến mà mlkit chưa có. Các thư viện cùng
loại chọn một trong hai hướng:
- mô hình thống kê riêng cho từng chuỗi (ARIMA, ETS);
- hồi quy "toàn cục" trên đặc trưng lag, dùng chung cho nhiều chuỗi (AutoGluon-TimeSeries, mlforecast).

Hướng thứ hai tái dùng được toàn bộ pipeline hiện có: QC, tiền xử lý, ModelArena, bundle và giám sát.

## Quyết định
- `task_type: forecasting` là bí danh: cấu hình được chuẩn hóa thành `task_type: regression` cộng với mục
  `forecasting: {horizon, season, lags, windows}`. Không thêm nhánh `task_type` mới vào các tầng phía dưới.
- **Chiến lược direct.** Mọi đặc trưng của dòng `t` chỉ dùng target tại `t - horizon` trở về trước:
  - lag;
  - rolling mean/std của chuỗi đã dịch `horizon`;
  - giá trị seasonal naive.

  Một mô hình vì thế dự báo trước đúng `horizon` kỳ. `lags < horizon` bị từ chối.
- `season` tự dò bằng đỉnh ACF khi không khai báo.
- `group_col` là mã chuỗi, dùng một mô hình chung cho mọi chuỗi. Riêng với forecasting, CV **không** dùng
  GroupKFold. Thời gian quyết định cách chia (TimeSeriesSplit trong Arena, holdout theo thời gian), để không học
  từ "tương lai" của chuỗi khác.
- QC bỏ qua kiểm tra rò rỉ cho các cột `<target>__*` do mlkit tạo ra. Các cột này tương quan mạnh với target theo
  thiết kế, nhưng chỉ chứa thông tin quá khứ.
- Baseline bắt buộc là **seasonal naive**. Mỗi lần train ghi:
  - `holdout_forecast` (MAE, RMSE, sMAPE, MASE);
  - `seasonal_naive`;
  - `beats_seasonal_naive`.

  MASE được chia theo MAE của seasonal naive trong tập train, nên vẫn đúng khi gộp nhiều chuỗi.
- Khi dự báo, người dùng truyền lịch sử (có target) nối với các kỳ cần dự báo (target trống). Tham số đặc trưng thực
  dùng (season, lags, windows) được lưu trong bundle để dựng lại y hệt lúc train.

## Hệ quả
- Không cần phụ thuộc mới (không dùng statsmodels; ACF và phân rã cổ điển được viết bằng numpy/pandas).
- Benchmark `--suite timeseries` gồm 3 bộ tổng hợp offline (CI) và 2 bộ OpenML. Chỉ số đánh giá là
  `skill = 1 - MAE / MAE(seasonal naive)`.
- Giới hạn đã biết:
  - giả định các kỳ cách đều nhau trong mỗi chuỗi (không tự lấp kỳ thiếu);
  - dự báo đa bước dùng một mô hình cho `horizon` lớn nhất, chưa có mô hình riêng cho từng bước;
  - chưa có khoảng dự báo (prediction interval).
