# ADR-001: BaseFrameTransformer kế thừa sklearn BaseEstimator và TransformerMixin

## Trạng thái
Đã chốt

## Bối cảnh
Cần quyết định liệu `BaseFrameTransformer` nên wrap hay kế thừa từ scikit-learn.

## Quyết định
`BaseFrameTransformer` kế thừa trực tiếp từ `sklearn.base.BaseEstimator` và `sklearn.base.TransformerMixin`.

## Hệ quả
- Khả năng tương thích 100% với `sklearn.pipeline.Pipeline`, `sklearn.model_selection.GridSearchCV`.
- Hỗ trợ clone, set_params, get_params chuẩn sklearn.
- Lưu ý: Xử lý linh hoạt cả DataFrame và numpy array ở biên ngoài khi nhận đầu vào.
