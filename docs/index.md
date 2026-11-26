# mlkit

Bộ công cụ học máy cho dữ liệu dạng bảng. Package trên PyPI tên là `mltk-package`; khi dùng trong code thì
`import mlkit`.

```bash
pip install mltk-package            # lõi: pandas, scikit-learn, LightGBM
pip install "mltk-package[all]"     # thêm biểu đồ, SHAP, Optuna, XGBoost, REST API, Parquet/Excel
```

## Bao trùm những gì

| Công đoạn | Năng lực chính |
|---|---|
| Dữ liệu | Nhận diện kiểu ngữ nghĩa của từng cột. QC phát hiện rò rỉ, giá trị thiếu đội lốt, cột trùng, mất cân bằng. EDA kèm 20+ biểu đồ và báo cáo HTML. Dữ liệu > 1 triệu dòng được lấy mẫu, nhưng các phép đếm vẫn chính xác |
| Tiền xử lý và đặc trưng | Chuyển kiểu, điền thiếu, mã hóa (gồm target encoding theo lớp), ngày tháng, văn bản theo nghĩa (TF-IDF/embedding), nối bảng sự kiện đúng thời điểm |
| Mô hình | `ModelArena` so sánh nhiều họ mô hình bằng CV ở mức pipeline (không rò rỉ), tối ưu Optuna, ensemble vote/stack, hiệu chuẩn, chọn ngưỡng; preset `fast` / `balanced` / `best_quality` với ngân sách thời gian |
| Loại bài toán | Phân loại nhị phân, nhiều lớp, hồi quy, dự báo chuỗi thời gian (so với seasonal naive) |
| Sau huấn luyện | Bundle có checksum và chữ ký HMAC, dự đoán theo khối, giải thích từng dự đoán, theo dõi drift, REST API, MLflow |

## Bắt đầu nhanh

```python
from mlkit import TaskConfig
from mlkit.runner import read_table, train, predict_bundle

df = read_table("data.csv")
res = train(TaskConfig(target="churn", modelling={"preset": "balanced"}), df, output_dir="bundle")
print(res.metrics["champion"], res.metrics["holdout"])
preds = predict_bundle("bundle", df.drop(columns="churn"), explain=3)
```

Hoặc dùng dòng lệnh: `mlkit run config.yaml --out bundle`, `mlkit predict bundle new.csv --out preds.parquet --chunksize 200000`.

Mỗi công đoạn có hàm nào, dùng để làm gì: xem [Hướng dẫn sử dụng](HUONG_DAN_SU_DUNG.md). Kế hoạch nâng cấp định kỳ
và cam kết về độ ổn định của API: xem [Lộ trình](ROADMAP.md).
