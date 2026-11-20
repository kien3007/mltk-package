# Kiến trúc hệ thống: bộ package cho các bước build model

*Bản nháp v0.6, 20/11/2026. Dùng để thảo luận trong team trước khi viết code.*

> [!NOTE]
> v0.6 so với v0.5: bổ sung constraints từ requirements thực tế — dữ liệu đầu vào là CSV; chỉ numeric và categorical (không có text/image); target column do DS truyền vào, không assume; thêm `read_csv()` vào mltk-qc; `CategoricalEncoder` auto-switch theo cardinality; high-cardinality strategy cần benchmark trước khi chốt default.

---

## Mục lục

1. [Mục tiêu, phạm vi và nguyên tắc](#1-mục-tiêu-phạm-vi-và-nguyên-tắc)
2. [Kiến trúc tổng thể](#2-kiến-trúc-tổng-thể)
3. [Package core](#3-package-core)
4. [Thiết kế từng package](#4-thiết-kế-từng-package)
5. [Vòng đời artifact và khả năng tái lập](#5-vòng-đời-artifact-và-khả-năng-tái-lập)
6. [Cấu trúc repo và quản lý package](#6-cấu-trúc-repo-và-quản-lý-package)
7. [Chất lượng, test và CI/CD](#7-chất-lượng-test-và-cicd)
8. [Lộ trình triển khai](#8-lộ-trình-triển-khai)
9. [Architecture Decision Records (ADR)](#9-architecture-decision-records-adr)
10. [Quyết định đã chốt và câu hỏi mở](#10-quyết-định-đã-chốt-và-câu-hỏi-mở)
11. [Rủi ro và cách giảm thiểu](#11-rủi-ro-và-cách-giảm-thiểu)

---

## 1. Mục tiêu, phạm vi và nguyên tắc

**Mục tiêu.** Xây dựng một **bộ thư viện Python** (library collection, không phải framework) — mỗi bước trong quy trình build model là một package riêng: QC & EDA, Preprocessing, Feature engineering, Feature selection, Modelling, Monitoring.

**Người dùng mục tiêu: Data Scientist** làm việc trong notebook hoặc script Python. Họ import các class cụ thể, dùng ngay, không cần học convention nội bộ hay subclass bất cứ thứ gì. Khi cần thứ gì đó mltk chưa có, họ tự viết class Python thông thường — không bắt buộc kế thừa base class.

Các package dùng chung một lõi (`core`) để đảm bảo nhất quán nội bộ về logging, schema, và artifact — nhưng đây là chi tiết **implementation**, không phải API mà DS phải biết.

### Trong phạm vi v1

- Backend dữ liệu duy nhất là pandas, dữ liệu vừa trong RAM của một máy.
- **Định dạng đầu vào: CSV** — mltk-qc cung cấp `read_csv()` xử lý encoding, separator, dtype inference.
- **Kiểu dữ liệu: chỉ numeric (int/float) và categorical (object/category/bool)** — không có text, image, hay time series trong v1.
- **Target column: do DS truyền vào tường minh** — không bao giờ assume tên cột; `X` không chứa target.
- Bài toán tabular: **phân loại nhị phân và hồi quy** (chuỗi thời gian để v2).
- Chạy batch (offline); tool là thư viện, không phải nền tảng.
- **Tương thích nhiều dataset**: transformer tự detect kiểu cột, không assume tên cột; hoạt động với bất kỳ số cột nào.

### Ngoài phạm vi v1

- Spark, huấn luyện phân tán, đa backend DataFrame.
- Serving online (API real-time), feature store, giao diện web.
- Orchestration không bắt buộc; ví dụ tích hợp dùng **Prefect** (`prefect-mlflow` plugin). Kedro không phù hợp vì xung đột pipeline concept với `sklearn.Pipeline` của mltk.
- CV theo thời gian (TimeSeriesSplit) và CV theo nhóm: để v2.
- Xuất sang ONNX hoặc MLflow pyfunc: để v2.
- Text, NLP, image: ngoài phạm vi.
- High-cardinality default strategy: cần benchmark trên data thực — tạm thời DS tự chọn, chốt default ở v1.1.

### Nguyên tắc thiết kế

1. **Độc lập module**: mỗi bước là một package, chỉ phụ thuộc `core`, không phụ thuộc lẫn nhau.
2. **Library, không phải framework**: DS import class cụ thể và dùng ngay; không bắt buộc kế thừa base class hay khai báo YAML để bắt đầu. Base class là chi tiết nội bộ dành cho contributor của mltk.
3. **pandas vào, pandas ra**: chỉ chuyển đổi định dạng ở biên khi gọi thư viện ngoài (numpy array, scipy sparse).
4. **Cấu hình linh hoạt**: tham số truyền qua `__init__` (primary) hoặc YAML + pydantic-settings (optional, dành cho automation/CI); không hard-code tham số trong code.
5. **Tái lập được**: cố định `random_state`, ghi lại phiên bản gói, hash dữ liệu, git SHA.
6. **Chất lượng tự động hóa**: CI bắt buộc; lỗi phải sớm và rõ ràng.
7. **MVP trước, mở rộng sau**: mỗi giai đoạn có tiêu chí hoàn thành rõ ràng.

---

## 2. Kiến trúc tổng thể

### Sơ đồ phụ thuộc

Các package bước đều phụ thuộc vào `core` (mũi tên chỉ hướng phụ thuộc). Meta-package `mltk` là tùy chọn để cài tất cả cùng lúc.

```
mltk-qc            --\
mltk-preprocessing --\
mltk-features      ---+--> mltk-core
mltk-selection     ---+
mltk-modelling     ---+
mltk-monitoring    --/

mltk (meta)  -->  tất cả package trên  [optional]
```

> [!IMPORTANT]
> Không có phụ thuộc ngang (ví dụ `mltk-features` không được import từ `mltk-preprocessing`). Nếu cần dùng chung logic, đưa vào `mltk-core`.

### Luồng dữ liệu end-to-end

```
raw data
  │
  ▼
[mltk-qc]           ─── QC Report (JSON/HTML)
  │
  ▼
[mltk-preprocessing] ─── fitted transformers
  │
  ▼
[mltk-features]      ─── fitted transformers
  │
  ▼
[mltk-selection]     ─── feature list + selection report
  │
  ▼
[mltk-modelling]     ─── model, metrics, model card → MLflow Registry
  │
  ▼
[mltk-monitoring]    ─── drift report + alert flags
  (nạp reference từ artifact của bước modelling)
```

### Bảng tóm tắt package

> [!NOTE]
> Cột "DS dùng gì" là public API mà Data Scientist thực sự import và gọi. `BaseFrameTransformer` / `BaseMonitor` là **internal** — không xuất hiện trong docs dành cho DS.

| Package | Trách nhiệm | DS dùng gì | Thư viện chính | Đầu ra |
|---|---|---|---|---|
| `mltk-core` | Internal: logging, schema, artifact, base class | *(không dùng trực tiếp)* | pandas, scikit-learn, pandera, pydantic-settings | Thư viện nội bộ |
| `mltk-qc` | Kiểm tra chất lượng và EDA | `run_qc(df)`, `QCReport` | pandas, pandera | Report (JSON/HTML) |
| `mltk-preprocessing` | Làm sạch, mã hóa, chuẩn hóa | `MissingImputer`, `CategoricalEncoder`, `OutlierClipper`, `FeatureScaler`, ... | scikit-learn, category\_encoders | Transformer đã fit |
| `mltk-features` | Tạo đặc trưng mới | `DatetimeFeatures`, `RatioFeatures`, `GroupAggFeatures`, `TargetEncoder`, ... | feature-engine | Transformer đã fit |
| `mltk-selection` | Chọn đặc trưng | `VarianceFilter`, `CorrelationFilter`, `SHAPSelector`, `StabilityFilter`, ... | scikit-learn, shap | DataFrame đã lọc + report |
| `mltk-modelling` | Huấn luyện, CV, tuning, đánh giá | `LGBMTrainer`, `XGBTrainer`, `CVEvaluator`, `OptunaSearch`, ... | LightGBM/XGBoost, Optuna, MLflow | Model, metrics, model card |
| `mltk-monitoring` | Phát hiện drift và suy giảm | `DataDriftMonitor`, `PredictionDriftMonitor`, `PerformanceMonitor`, ... | Evidently (qua adapter) | Report và cảnh báo |

---

## 3. Package core

> [!NOTE]
> Mục này dành cho **contributor nội bộ** của mltk, không phải Data Scientist. DS không cần đọc mục này để dùng thư viện.

### 3.1 Các thành phần

#### Check / CheckResult / Report

```python
# Sketch — không phải code production
class CheckResult:
    name: str
    level: Literal["INFO", "WARNING", "ERROR"]
    passed: bool
    details: dict

class Check(ABC):
    @abstractmethod
    def run(self, df: pd.DataFrame) -> CheckResult: ...

class Report:
    results: list[CheckResult]

    @property
    def passed(self) -> bool:  # True nếu không có ERROR
        ...

    def to_frame(self) -> pd.DataFrame: ...
    def to_json(self) -> str: ...
    def to_html(self) -> str: ...      # Mục 4.1 — ưu tiên cả hai format
    def raise_if_failed(self) -> None: ...  # dùng khi fail_on=ERROR
```

#### BaseFrameTransformer — chi tiết nội bộ

> [!IMPORTANT]
> `BaseFrameTransformer` là **internal implementation detail**. DS không cần biết đến nó. Nó chỉ dùng để contributor nội bộ viết thêm component mới cho mltk một cách nhất quán, và để đảm bảo tất cả component đều tương thích `sklearn.Pipeline` tự động.

```python
from sklearn.base import BaseEstimator, TransformerMixin

class BaseFrameTransformer(BaseEstimator, TransformerMixin, ABC):
    """
    - fit(X, y=None)  → self
    - transform(X)    → pd.DataFrame  (KHÔNG sửa X tại chỗ)
    - fit_transform    được kế thừa từ TransformerMixin

    Lợi ích:
    - Lắp trực tiếp vào sklearn.Pipeline mà không cần wrapper.
    - set_params() / get_params() hoạt động → Optuna và GridSearch dùng được.
    - clone() hoạt động → cross-validation an toàn.
    """
    def fit(self, X: pd.DataFrame, y=None) -> "BaseFrameTransformer":
        self._validate_input(X)
        self._feature_names_in = list(X.columns)
        return self._fit(X, y)

    @abstractmethod
    def _fit(self, X: pd.DataFrame, y=None) -> "BaseFrameTransformer": ...

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        self._validate_input(X)
        result = self._transform(X.copy())   # luôn copy để không sửa đầu vào
        assert isinstance(result, pd.DataFrame)
        return result

    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame: ...

    def get_feature_names_out(self) -> list[str]: ...
    def _validate_input(self, X: pd.DataFrame) -> None: ...  # kiểm tra cột và kiểu
```

**Lưu ý chuyển đổi biên**: khi thư viện bên ngoài yêu cầu numpy array, chỉ chuyển đổi bên trong `_fit` / `_transform` và khôi phục về DataFrame trước khi trả về.

#### BaseMonitor

```python
class BaseMonitor(ABC):
    def fit(self, reference: pd.DataFrame) -> "BaseMonitor":
        """Học thống kê tham chiếu từ tập train."""
        ...

    @abstractmethod
    def evaluate(self, current: pd.DataFrame) -> Report: ...
```

#### Data contract (Pandera)

```python
# Đề xuất — cần team đồng ý
from mltk_core.contract import define_schema, ContractViolation

InputSchema = define_schema(
    columns={"age": pa.Int, "income": pa.Float, "label": pa.Int},
    nullable={"income": True},
)
# Schema được lưu cùng artifact; kiểm tra lại khi serving và monitoring.
```

#### Config (pydantic-settings)

Dùng `pydantic-settings.BaseSettings` thay vì `pydantic.BaseModel` để hỗ trợ override theo tầng:

```
env var  >  .env file  >  config.yaml  >  default trong class
```

```python
# pydantic-settings tự đọc env var và .env file;
# YAML được nạp thủ công vào model_config customise_sources.
from pydantic_settings import BaseSettings, YamlConfigSettingsSource

class PreprocessingConfig(BaseSettings):
    model_config = SettingsConfigDict(
        frozen=True,
        env_prefix="MLTK_PREP_",   # MLTK_PREP_IMPUTE_STRATEGY override từ env
        yaml_file="config.yaml",
    )
    impute_strategy: Literal["mean", "median", "most_frequent"] = "median"
    outlier_method: Literal["clip", "winsorize"] = "clip"
    outlier_quantile: float = Field(0.01, ge=0.0, le=0.5)

    @classmethod
    def settings_customise_sources(cls, settings_cls, **kwargs):
        return (
            kwargs["env_settings"],          # 1. env var (ưu tiên cao nhất)
            kwargs["dotenv_settings"],       # 2. .env file
            YamlConfigSettingsSource(settings_cls),  # 3. config.yaml
            kwargs["init_settings"],         # 4. giá trị truyền tường minh
        )

# Nạp — tự động đọc env var, .env, và YAML:
cfg = PreprocessingConfig()

# CI/CD chỉ cần set env var, không cần sửa file:
# MLTK_PREP_IMPUTE_STRATEGY=mean python train.py
```

> [!TIP]
> Mỗi package bước có class config riêng với `env_prefix` riêng (ví dụ `MLTK_QC_`, `MLTK_MODEL_`). Tất cả kế thừa `MltkBaseSettings` từ `mltk-core` để dùng chung prefix root và cấu hình `yaml_file` mặc định.

#### Logging và exceptions

**Format log**: structured JSON (một dòng mỗi event) để dễ ingest vào ELK / Cloud Logging. Mỗi log entry có các trường bắt buộc:

```json
{
  "ts": "2026-11-20T13:00:00Z",
  "level": "INFO",
  "logger": "mltk.preprocessing",
  "run_id": "abc123",
  "step": "MissingImputer.fit",
  "msg": "Fit completed",
  "extra": {}
}
```

`run_id` là UUID được tạo lúc bắt đầu pipeline và truyền qua context var — không cần truyền tường minh vào từng hàm.

**Custom exceptions**:

```python
class SchemaError(MltkError): ...      # schema / kiểu dữ liệu không khớp
class NotFittedError(MltkError): ...   # gọi transform trước khi fit
class ContractViolation(MltkError): .. # vi phạm data contract
class LeakageError(MltkError): ...     # phát hiện rò rỉ trong test
```

#### Metadata artifact

Mọi artifact lưu ra đều kèm:

```python
@dataclass
class ArtifactMetadata:
    created_at: datetime
    mltk_core_version: str
    package_versions: dict[str, str]   # toàn bộ packages trong venv
    git_sha: str
    data_hash: str                     # SHA-256 của DataFrame đầu vào
    random_state: int | None
    run_id: str
```

### 3.2 Quy ước chung

- `random_state` luôn truyền được qua `__init__` và lưu vào `self.random_state`.
- Không sửa DataFrame đầu vào tại chỗ: `_transform` nhận bản copy, `_fit` không được sửa X.
- Tên cột là `str`; kiểu dữ liệu khai báo trong Pandera schema.
- Hàm public không nhận `*args`; dùng keyword-only argument cho tham số tùy chọn.

---

## 4. Thiết kế từng package

### 4.1 mltk-qc (QC & EDA)

**`read_csv()` — điểm vào chuẩn cho mọi dataset CSV:**

```python
from mltk_qc import read_csv

df = read_csv("data/customers.csv")
# Tự handle:
# - Encoding detection (UTF-8, latin-1, ...)
# - Separator detection (comma, semicolon, tab)
# - Dtype inference: "1.5" → float, "2024-01-01" → datetime, "true" → bool
# - Báo ngay nếu file rỗng, header lỗi, hoặc toàn bộ cột là string khi không nên
```

**Convention bắt buộc — tách X và y trước khi truyền vào mltk:**

```python
X = df.drop(columns=["ten_cot_target"])   # DS truyền tên cột target
y = df["ten_cot_target"]
# Từ đây: tất cả mltk API nhận X (không có target) và y riêng biệt
```

**Column type inference — dùng chung toàn bộ mltk:**

```python
# Internal utility của mltk-core — DS không gọi trực tiếp
# Tất cả transformer dùng để tự detect cột
#   NUMERIC    : dtype int, float
#   CATEGORICAL: dtype object, category, bool
# Datetime được detect nhưng không phải focus của v1
```

**Check v1:**

| Nhóm | Check cụ thể |
|---|---|
| Schema | Kiểu dữ liệu (chỉ numeric / categorical), cột bắt buộc |
| Completeness | Tỷ lệ thiếu theo cột và theo dòng |
| Uniqueness | Dòng trùng, trùng theo khóa nghiệp vụ |
| Range | Khoảng giá trị số; số giá trị phân biệt cho categorical |
| Constant | Cột hằng (variance = 0 hoặc 1 giá trị duy nhất) |
| High-cardinality | Cảnh báo cột categorical có cardinality > ngưỡng (WARNING, không block) |
| Target | Phân bố biến mục tiêu, mất cân bằng lớp |
| Leakage | Tương quan gần tuyệt đối với target (Spearman > 0.99) |

**EDA:** thống kê mô tả, phân bố, tương quan, tóm tắt theo target. `ydata-profiling` là optional extra (`mltk-qc[profiling]`) vì kích thước lớn.

**Đầu ra:** Report có thể chặn pipeline (`report.raise_if_failed()`) hoặc chỉ cảnh báo. Hỗ trợ cả JSON và HTML.

### 4.2 mltk-preprocessing

Thành phần v1 — tất cả tự detect kiểu cột, không cần DS chỉ định columns:

| Transformer | Mô tả |
|---|---|
| `TypeCaster` | Ép kiểu theo schema khai báo; bước đầu tiên nên luôn có |
| `MissingImputer` | Tự detect: median cho numeric, mode cho categorical; học trên train |
| `OutlierClipper` | Chỉ apply trên numeric tự động; categorical bị bỏ qua |
| `RareCategoryGrouper` | Gộp category hiếm (tần suất < ngưỡng) thành `"__rare__"` |
| `CategoricalEncoder` | Auto-switch encoding theo cardinality (xem bên dưới) |
| `FeatureScaler` | StandardScaler / RobustScaler / MinMaxScaler trả về DataFrame |

**`CategoricalEncoder` — auto-switch theo cardinality:**

```python
encoder = CategoricalEncoder(
    # Ngưỡng cardinality (configurable, cần benchmark để chốt default)
    low_card_threshold=10,      # < 10 unique  → one-hot
    high_card_threshold=50,     # > 50 unique  → target encoding (cross-fitting)
    # Giữa 10–50: ordinal encoding (default cho đến khi có benchmark)
    mid_strategy="ordinal",

    # Override cho cột cụ thể nếu DS muốn
    column_overrides={"product_id": "frequency"},

    cv=5,           # số fold cho cross-fitting của target encoding
    random_state=42,
)
```

> [!WARNING]
> **High-cardinality strategy chưa được chốt default** — cần benchmark trên dataset thực trước khi quyết định. DS nên thử nhiều strategy và so sánh AUC trong giai đoạn thực nghiệm. Default tạm thời là `"ordinal"` cho mid-cardinality và `"target"` cho high-cardinality; sẽ cập nhật ở v1.1 sau khi có kết quả benchmark.

### 4.3 mltk-features

Thành phần v1:

| Transformer | Mô tả |
|---|---|
| `DatetimeFeatureExtractor` | year, month, dayofweek, hour, is_weekend, v.v. |
| `RatioFeatureCreator` | Tỷ lệ và hiệu giữa các cặp cột số theo khai báo |
| `GroupAggregator` | mean/std/min/max/count theo nhóm; học trên train |
| `TargetEncoder` | Encoding theo target với cross-fitting tránh rò rỉ |

> [!CAUTION]
> **Quy tắc chống rò rỉ (bắt buộc):** mọi phép học có sử dụng `y` chỉ được `fit` trên tập train. `TargetEncoder` và `GroupAggregator` với tổng hợp có target phải dùng cross-fitting. Mỗi transformer này phải đi kèm test kiểm tra rò rỉ (xem Mục 7).

### 4.4 mltk-selection

Thành phần v1 — `FeatureSelector` kế thừa `BaseFrameTransformer` (trả về DataFrame với các cột đã chọn):

| Selector | Phương pháp |
|---|---|
| `VarianceFilter` | Loại cột phương sai thấp |
| `MissingRateFilter` | Loại cột thiếu > ngưỡng |
| `CorrelationFilter` | Loại cột tương quan cao (Pearson / Spearman) |
| `ModelImportanceSelector` | Chọn theo feature importance của model đơn giản |
| `SHAPSelector` | Chọn theo SHAP mean absolute value |
| `StabilityFilter` | Loại cột PSI cao giữa train và validation (dấu hiệu rò rỉ hoặc không ổn định) |

**Đầu ra:** DataFrame đã lọc + artifact chứa danh sách feature được chọn, lý do loại bỏ cho từng cột bị loại.

### 4.5 mltk-modelling

**Wrapper model:**
- `LGBMWrapper`, `XGBWrapper`, `SklearnModelWrapper` — tuân thủ sklearn Estimator API.
- Nhận `pd.DataFrame` đầu vào (không phải numpy) nhờ `BaseEstimator`.

**CV strategy (v1):** StratifiedKFold (phân loại), KFold (hồi quy). TimeSeriesSplit và GroupKFold để v2.

**Tuning:** Optuna với pruning (MedianPruner). Số trial và timeout cấu hình qua YAML.

**Đánh giá:**
- Phân loại: AUC-ROC, AUC-PR, F1, calibration curve, threshold analysis.
- Hồi quy: RMSE, MAE, R², residual plot.

**MLflow integration:**
- Mọi lần chạy log: params, metrics, feature list, input schema, artifact metadata, phiên bản packages.
- Lưu pipeline đầy đủ (transformers + model) bằng **skops** (ưu tiên) hoặc joblib kèm version check khi load.
- Đăng ký vào MLflow Model Registry.

**Model card**: YAML ghi thông tin bài toán, dữ liệu, metrics, giới hạn sử dụng. Lưu cùng artifact.

### 4.6 mltk-monitoring

**Các loại kiểm tra:**

| Loại | Phương pháp |
|---|---|
| Data drift (số) | PSI, KS test |
| Data drift (categorical) | Chi-square, PSI |
| Prediction drift | PSI trên phân bố dự đoán |
| Performance degradation | Metrics so với baseline (khi có nhãn thực) |
| Data quality | Tỷ lệ thiếu, range, schema — tái dùng mltk-qc |

**Reference:** nạp từ artifact của model (thống kê tham chiếu được lưu lúc fit trong mltk-modelling).

**Evidently adapter:**

```python
# Evidently được bọc sau adapter; phiên bản được pin trong pyproject.toml
# Nếu Evidently đổi API, chỉ sửa adapter, không ảnh hưởng người dùng.
class EvidentlyDriftAdapter:
    EVIDENTLY_VERSION = "0.4.x"   # pin cụ thể
    ...
```

**Đầu ra:** Report và cờ cảnh báo (`alert_level`: INFO / WARNING / CRITICAL) theo ngưỡng cấu hình.

---

## 5. Vòng đời artifact và khả năng tái lập

### Luồng artifact

**Primary workflow của DS** — viết Python code, lưu artifact thủ công hoặc qua helper:

```python
# DS làm trong notebook — đây là primary workflow
from mltk_preprocessing import MissingImputer, CategoricalEncoder
from mltk_selection import SHAPSelector
from mltk_modelling import LGBMTrainer, save_artifact

imputer  = MissingImputer(strategy="median")
encoder  = CategoricalEncoder(method="onehot")
selector = SHAPSelector(threshold=0.01)
trainer  = LGBMTrainer(n_estimators=500, random_state=42)

X_clean = imputer.fit_transform(X_train)
X_enc   = encoder.fit_transform(X_clean)
X_sel   = selector.fit_transform(X_enc, y_train)
trainer.fit(X_sel, y_train)

# Lưu toàn bộ artifact một lệnh
save_artifact(
    steps=[imputer, encoder, selector, trainer],
    output_dir="./artifacts/run_001",
)
```

**Artifact được lưu:**

```
artifacts/run_001/
  ├── pipeline.skops              # các bước đã fit
  ├── input_schema.json           # Pandera schema
  ├── selected_features.json
  ├── reference_stats.json        # cho monitoring
  ├── metrics.json
  ├── model_card.yaml
  └── artifact_metadata.json      # git SHA, hash, versions
```

**Optional — dùng sklearn.Pipeline** nếu DS muốn gọn hơn:

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("impute",  MissingImputer(strategy="median")),
    ("encode",  CategoricalEncoder(method="onehot")),
    ("select",  SHAPSelector(threshold=0.01)),
    ("model",   LGBMTrainer(n_estimators=500, random_state=42)),
])
pipe.fit(X_train, y_train)
save_artifact(pipeline=pipe, output_dir="./artifacts/run_001")
```

**Optional — MLflow logging** (tự động khi DS gọi `save_artifact`):

```
artifact_metadata.json → MLflow run params + tags
metrics.json           → MLflow metrics
pipeline.skops         → MLflow artifact
model_card.yaml        → MLflow artifact
→ Đăng ký vào MLflow Model Registry nếu DS muốn
```

### Tái lập

| Yếu tố | Cách xử lý |
|---|---|
| Random seed | `random_state` truyền tường minh; không dùng global seed |
| Phiên bản code | `git_sha` ghi vào metadata |
| Phiên bản thư viện | `package_versions` (toàn bộ venv) ghi vào metadata |
| Dữ liệu đầu vào | SHA-256 hash ghi vào metadata; dữ liệu lớn dùng DVC (tùy chọn) |
| Serialization | Dùng skops; khi load kiểm tra `sklearn` và `pandas` version; **không load pickle từ nguồn không tin cậy** |

---

## 6. Cấu trúc repo và quản lý package

### Cấu trúc thư mục

```
mltk/
  pyproject.toml              # root của uv workspace
  uv.lock                     # lock file chung
  packages/
    mltk-core/
      pyproject.toml
      src/mltk_core/
        base.py               # BaseFrameTransformer, BaseMonitor
        check.py              # Check, CheckResult, Report
        contract.py           # Pandera helpers
        config.py             # pydantic base config
        logging.py            # structured JSON logger, run_id context var
        exceptions.py
        metadata.py           # ArtifactMetadata
    mltk-qc/
    mltk-preprocessing/
    mltk-features/
    mltk-selection/
    mltk-modelling/
    mltk-monitoring/
  docs/                       # MkDocs Material + mkdocstrings
  examples/
    basic_pipeline.py         # pipeline end-to-end trên dữ liệu mẫu nhỏ
    notebooks/                # notebook ví dụ (kiểm tra bằng nbmake trong CI)
  .github/
    workflows/
      ci.yaml
      release.yaml
      docs.yaml
  .pre-commit-config.yaml
  ADRs/
    ADR-001-basframetransformer-sklearn.md
    ADR-002-logging-format.md
    ADR-003-serialization.md
```

### Workspace uv

- Một `uv.lock` chung cho toàn repo; mỗi package có `pyproject.toml` riêng.
- Package bước khai báo phụ thuộc `mltk-core` theo workspace path.
- Thư viện nặng là optional extra: `mltk-qc[profiling]`, `mltk-selection[shap]`, `mltk-monitoring[evidently]`.

### Naming

- **Tên import phẳng cho v1**: `mltk_core`, `mltk_qc`, `mltk_preprocessing`, v.v.
- **Namespace package `mltk.*` cho v2** (đã chốt): khi số package tăng, chuyển sang namespace package để import dạng `mltk.core`, `mltk.qc`.

### Versioning và coupling liên package

> [!IMPORTANT]
> **Chiến lược versioning liên package:**
> - Mỗi package dùng SemVer riêng. Tag dạng `mltk-core-v1.2.0`.
> - `core` công bố **compatibility matrix**: phiên bản core nào tương thích với phiên bản nào của package bước.
> - Khi `core` có breaking change (major bump), tất cả package bước phải release major cùng lúc (coordinated release). Quy trình: tạo branch `breaking/core-v2`, cập nhật tất cả package, merge và tag cùng ngày.
> - Package bước chỉ khai báo `mltk-core >= X.0, < (X+1).0` để tránh auto-upgrade qua breaking change.
> - Changelog và release tự động bằng **release-please** (đã chốt).

### Phát hành

Publish lên registry nội bộ (team cần xác định — xem [Câu hỏi mở #2](#10-quyết-định-đã-chốt-và-câu-hỏi-mở)).

---

## 7. Chất lượng, test và CI/CD

### Chiến lược test

#### Unit test
Mỗi thành phần có unit test riêng. Dữ liệu mẫu tạo bằng fixtures, không dùng file lớn.

#### Contract test dùng chung (cơ chế cụ thể)

Đặt trong `mltk-core/tests/contract/`. Mọi transformer đều phải pass bộ test này:

```python
# mltk-core/tests/contract/conftest.py
def pytest_generate_tests(metafunc):
    if "transformer_cls" in metafunc.fixturenames:
        # Mỗi package đăng ký transformer của mình vào registry này
        metafunc.parametrize("transformer_cls", TRANSFORMER_REGISTRY)

# mltk-core/tests/contract/test_transformer_contract.py
def test_fit_transform_returns_dataframe(transformer_cls, sample_df):
    t = transformer_cls()
    result = t.fit_transform(sample_df)
    assert isinstance(result, pd.DataFrame)

def test_does_not_mutate_input(transformer_cls, sample_df):
    original = sample_df.copy()
    transformer_cls().fit_transform(sample_df)
    pd.testing.assert_frame_equal(sample_df, original)

def test_raises_not_fitted_error(transformer_cls, sample_df):
    with pytest.raises(NotFittedError):
        transformer_cls().transform(sample_df)

def test_missing_column_raises_schema_error(transformer_cls, sample_df):
    t = transformer_cls().fit(sample_df)
    with pytest.raises(SchemaError):
        t.transform(sample_df.drop(columns=[sample_df.columns[0]]))

def test_deterministic(transformer_cls, sample_df):
    r1 = transformer_cls(random_state=42).fit_transform(sample_df)
    r2 = transformer_cls(random_state=42).fit_transform(sample_df)
    pd.testing.assert_frame_equal(r1, r2)
```

**Cách đăng ký transformer mới**: trong `pyproject.toml` của package bước, khai báo entry point:

```toml
[project.entry-points."mltk.transformers"]
missing_imputer = "mltk_preprocessing.imputer:MissingImputer"
```

`mltk-core` đọc registry qua `importlib.metadata.entry_points` → contract test tự động bao gồm transformer mới khi cài package.

#### Test chống rò rỉ

```python
# Kiểm tra TargetEncoder không rò rỉ thông tin từ test set vào train
def test_no_leakage(encoder, train_df, test_df, y_train):
    enc = encoder.fit(train_df, y_train)
    # Shuffle target → AUC phải về ~0.5 nếu không có leakage
    y_shuffled = y_train.sample(frac=1).reset_index(drop=True)
    enc_shuffled = encoder.fit(train_df, y_shuffled)
    auc_original = roc_auc_score(y_train, enc.transform(train_df).values)
    auc_shuffled = roc_auc_score(y_train, enc_shuffled.transform(train_df).values)
    assert abs(auc_original - auc_shuffled) < LEAKAGE_THRESHOLD
```

#### Integration test

Pipeline end-to-end trên dataset mẫu nhỏ (< 1000 dòng, tổng < 5 giây). Chạy trong CI trên mỗi PR.

#### Notebook test

Notebook trong `examples/notebooks/` được kiểm tra bằng `nbmake` trong CI để tránh stale.

### Chất lượng mã

| Tool | Mục đích | Phạm vi |
|---|---|---|
| Ruff | Lint + format (thay flake8 + black + isort) | Toàn repo |
| mypy (strict) | Type check | `mltk-core`; nới dần cho package khác |
| pandas-stubs | Type stub cho pandas | Toàn repo |
| pre-commit | Chạy Ruff + mypy trước mỗi commit | Toàn repo |
| pip-audit | Kiểm tra CVE trong dependencies | CI |
| coverage (≥ 80%) | Ngưỡng coverage | CI, từng package |

### CI/CD với GitHub Actions

```
Pull request:
  ├── lint (Ruff)
  ├── type-check (mypy)
  ├── test (ma trận: Python 3.11 × 3.12)
  │     └── chỉ chạy package bị ảnh hưởng + package phụ thuộc vào nó
  ├── kiểm tra uv.lock còn khớp
  ├── pip-audit
  └── nbmake (nếu notebook thay đổi)

Nhánh chính (merge):
  └── build + deploy docs (MkDocs → GitHub Pages)

Tag release (mltk-{name}-v*):
  └── build + publish package tương ứng lên registry nội bộ
```

### Tài liệu

MkDocs Material + mkdocstrings (API sinh từ docstring). Mỗi package có:
- Trang hướng dẫn (Getting Started, How-to)
- API reference tự động
- Notebook ví dụ nhúng

---

## 8. Lộ trình triển khai

Thời lượng giả định cho team 2 đến 3 người; điều chỉnh theo thực tế.

| Giai đoạn | Thời lượng | Sản phẩm | Tiêu chí hoàn thành |
|---|---|---|---|
| **0. Chốt thiết kế** | Tuần 1 | ADR cho các quyết định chính, skeleton repo, CI chạy xanh, interface core đóng băng | Team đồng ý interface; CI xanh trên repo rỗng; ADR được merge |
| **1. MVP** | Tuần 2–3 | `mltk-core`, `mltk-qc`, `mltk-preprocessing`, ví dụ end-to-end | QC + preprocessing chạy được trên dữ liệu mẫu; contract test xanh; docs cơ bản |
| **2. Đặc trưng** | Tuần 4–5 | `mltk-features`, `mltk-selection` | Test chống rò rỉ xanh; selection xuất artifact đầy đủ |
| **3. Mô hình** | Tuần 6–7 | `mltk-modelling`, tích hợp MLflow | Huấn luyện, log và đăng ký model từ một file config; model card được tạo |
| **4. Giám sát & phát hành** | Tuần 8 | `mltk-monitoring`, hoàn thiện docs, release `0.1.0` | Pipeline end-to-end + monitoring chạy từ artifact; docs đầy đủ; nbmake xanh |

> [!TIP]
> Giai đoạn 0 quan trọng nhất: interface `core` sau khi đóng băng sẽ không được thay đổi backward-incompatible cho đến v2. Dành đủ thời gian review ở giai đoạn này.

---

## 9. Architecture Decision Records (ADR)

### ADR-001: BaseFrameTransformer kế thừa sklearn Estimator API

**Trạng thái:** Đã chốt

**Bối cảnh:** Cần quyết định `BaseFrameTransformer` wrap hay kế thừa sklearn.

**Quyết định:** Kế thừa `BaseEstimator + TransformerMixin`. Người dùng lắp trực tiếp vào `sklearn.Pipeline` mà không cần adapter.

**Hệ quả:**
- ✅ `set_params()`, `get_params()`, `clone()` hoạt động → Optuna và CV an toàn.
- ✅ Tương thích toàn bộ hệ sinh thái sklearn.
- ⚠️ `fit(X, y)` phải chấp nhận numpy array từ Pipeline — cần `_validate_input` phân biệt và chuyển đổi khi cần.

---

### ADR-002: Structured JSON logging với run_id

**Trạng thái:** Đã chốt

**Bối cảnh:** Cần format log thống nhất, dễ ingest vào hệ thống tập trung.

**Quyết định:** Structured JSON (một dòng mỗi event). `run_id` dùng `contextvars.ContextVar` — set một lần lúc bắt đầu pipeline, tự động xuất hiện trong mọi log entry mà không cần truyền tường minh.

**Hệ quả:**
- ✅ Có thể filter log theo run_id để debug một lần chạy cụ thể.
- ✅ Không thay đổi chữ ký hàm public.
- ⚠️ Người dùng dùng async cần chú ý ContextVar không tự copy qua thread pool.

---

### ADR-003: Serialization ưu tiên skops thay vì pickle

**Trạng thái:** Đã chốt

**Bối cảnh:** pickle không an toàn khi load từ nguồn ngoài; joblib cũng dựa trên pickle.

**Quyết định:** Dùng **skops** là mặc định. Fallback joblib chỉ khi skops chưa hỗ trợ custom class, kèm kiểm tra phiên bản sklearn và pandas khi load.

**Hệ quả:**
- ✅ Giảm rủi ro arbitrary code execution.
- ✅ Schema artifact có thể kiểm tra trước khi load.
- ⚠️ skops chưa hỗ trợ 100% custom class → cần test load artifact trên Python version khác trong CI.

---

### ADR-004: Bài toán v1 giới hạn ở phân loại nhị phân và hồi quy

**Trạng thái:** Đã chốt

**Bối cảnh:** Câu hỏi mở #3 — có cần chuỗi thời gian và CV theo thời gian ngay từ v1 không?

**Quyết định:** v1 chỉ hỗ trợ phân loại nhị phân và hồi quy. CV strategy: StratifiedKFold và KFold. TimeSeriesSplit, GroupKFold, và bài toán chuỗi thời gian để v2.

**Hệ quả:**
- ✅ Giảm scope, tập trung hoàn thiện nhanh hơn.
- ⚠️ Team có bài toán chuỗi thời gian ngay sẽ cần đợi v2 hoặc tự extend.

---

### ADR-005: mltk là library collection, không phải framework

**Trạng thái:** Đã chốt

**Bối cảnh:** Cần xác định rõ sản phẩm là gì để tránh thiết kế mâu thuẫn.

**Quyết định:**
- **Người dùng mục tiêu**: Data Scientist làm việc trong notebook/script Python.
- **Kiểu sản phẩm**: Library collection — DS import class cụ thể, dùng ngay, không cần học convention hay subclass bất cứ thứ gì.
- **Primary workflow**: Python code (notebook/script). YAML config là optional dành cho automation, không phải cách dùng chính.
- **Khi DS cần thứ chưa có**: tự viết class Python thông thường, không bắt buộc kế thừa `BaseFrameTransformer`.
- **`BaseFrameTransformer`**: internal implementation detail — dùng để contributor nội bộ viết thêm component nhất quán; **không** xuất hiện trong docs dành cho DS.

**Hệ quả:**
- ✅ DS có thể productive ngay sau khi đọc một ví dụ ngắn.
- ✅ Không có "magic" ẩn — DS thấy rõ dữ liệu chạy qua từng bước.
- ✅ Dễ debug: lỗi xảy ra ở đúng dòng code DS viết, không phải trong framework.
- ⚠️ Nhất quán phụ thuộc vào contributor nội bộ tuân thủ convention — contract test và CI bù đắp điều này.
- ⚠️ Docs cần tập trung vào class cụ thể và ví dụ notebook, không phải class diagram base class.

---

## 10. Quyết định đã chốt và câu hỏi mở

### Đã chốt

| Hạng mục | Quyết định |
|---|---|
| **Định nghĩa sản phẩm** | Library collection, không phải framework (ADR-005) |
| **Người dùng mục tiêu** | Data Scientist — dùng class cụ thể, không subclass, không học convention |
| **Primary workflow** | Python code (notebook/script); YAML là optional |
| Package manager | uv + pyproject.toml workspace |
| Estimator API | sklearn BaseEstimator + TransformerMixin — internal; DS không cần biết (ADR-001) |
| DataFrame backend | pandas duy nhất cho v1 |
| Config | pydantic-settings (BaseSettings, override env > .env > YAML > default) |
| Data contract | Pandera (schema khai báo, lưu cùng artifact) |
| Changelog & release | release-please (tag tự động, CHANGELOG.md sinh tự động) |
| Contract test | pytest parametrize qua `importlib.metadata.entry_points` |
| Namespace package | Tên import phẳng `mltk_*` cho v1; `mltk.*` namespace package cho v2 |
| Experiment tracking | MLflow |
| Monitoring library | Evidently qua adapter |
| Test framework | pytest + contract test dùng chung |
| Lint + format | Ruff + mypy |
| Docs | MkDocs Material + mkdocstrings — tập trung vào class cụ thể và notebook ví dụ |
| CI | GitHub Actions |
| Orchestration | Prefect (không bắt buộc — ví dụ trong `/examples`; `prefect-mlflow` plugin) |
| Logging format | Structured JSON + run_id qua ContextVar (ADR-002) |
| Serialization | skops ưu tiên, fallback joblib (ADR-003) |
| Bài toán v1 | Phân loại nhị phân + hồi quy (ADR-004) |

> [!NOTE]
> Không còn mục "Đề xuất cần team đồng ý" — tất cả đã được chốt.

### Câu hỏi mở còn lại

| # | Câu hỏi | Ảnh hưởng | Cần chốt trước |
|---|---|---|---|
| 1 | Tên bộ công cụ và quy ước đặt tên package | Naming toàn bộ codebase | Giai đoạn 0 |
| 2 | Registry nội bộ để publish package là gì? (PyPI private, Artifactory, GCP Artifact Registry, ...) | CI/CD release | Giai đoạn 0 |
| 3 | MLflow dùng server dùng chung hay chạy cục bộ? | Setup môi trường dev | Giai đoạn 3 |
| 4 | Phiên bản Python tối thiểu cần hỗ trợ (3.10, 3.11, hay 3.12+)? | Ma trận CI, type annotation syntax | Giai đoạn 0 |
| 5 | Report QC/EDA ưu tiên HTML, JSON hay cả hai? (đề xuất: cả hai) | mltk-qc interface | Giai đoạn 1 |
| 6 | **High-cardinality default strategy** — target encoding, frequency, hay ordinal? | `CategoricalEncoder` default | v1.1 sau khi có benchmark |

---

## 11. Rủi ro và cách giảm thiểu

| Rủi ro | Khả năng | Cách giảm thiểu |
|---|---|---|
| Làm quá nhiều thứ cùng lúc | Cao | MVP trước; mỗi giai đoạn có tiêu chí hoàn thành rõ; ADR giúp tránh tranh luận lại |
| Đổi interface core sau khi nhiều package đã dựa vào | Trung bình | Đóng băng interface ở Giai đoạn 0; contract test; SemVer + coordinated release |
| API Evidently thay đổi | Cao (Evidently hay đổi) | Pin version cụ thể trong pyproject.toml; bọc sau adapter |
| Rò rỉ dữ liệu trong feature engineering / selection | Trung bình | Chỉ fit trên train; cross-fitting; test chống rò rỉ bắt buộc trong CI |
| RAM không đủ khi dữ liệu lớn dần | Trung bình | Tối ưu dtype; đọc theo cột; interface thống nhất giúp đổi backend sau mà ít ảnh hưởng |
| Xung đột phụ thuộc giữa các package | Thấp | Một `uv.lock` chung; optional extras cho thư viện nặng |
| Nhiều người làm song song dễ lệch phong cách | Trung bình | ADR, code owners, pre-commit và CI bắt buộc |
| skops không hỗ trợ custom class | Thấp | Test load artifact trên Python version khác trong CI; fallback joblib có kiểm tra version |
| Notebook ví dụ bị stale | Cao (hay bị bỏ quên) | nbmake chạy trong CI; fail PR nếu notebook lỗi |
| Coordinated release bị quên | Trung bình | Checklist trong pull request template cho breaking changes; release-please tự động |
