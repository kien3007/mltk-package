"""High-level training / prediction / monitoring workflows used by the CLI and server.

These functions are the programmatic equivalent of ``mlkit run``, ``mlkit predict``
and ``mlkit monitor`` and can be called directly from Python.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# Importing the step modules populates the registry used by YAML-defined pipelines.
import mlkit.features
import mlkit.modelling
import mlkit.monitoring
import mlkit.preprocessing
import mlkit.qc_eda
import mlkit.selection  # noqa: F401
from mlkit.core.bundle import load_bundle
from mlkit.core.config import TaskConfig
from mlkit.core.errors import ValidationError
from mlkit.core.log import get_logger
from mlkit.core.pipeline import Pipeline
from mlkit.core.registry import get_step, list_steps
from mlkit.core.report import Report
from mlkit.features import DatetimeFeatures, TextFeatures
from mlkit.modelling import ModelArena, compute_bootstrap_ci, compute_metrics
from mlkit.modelling.evaluation import AVAILABLE_METRICS, LOWER_IS_BETTER
from mlkit.monitoring import DriftMonitor
from mlkit.preprocessing import (
    CategoricalEncoder,
    FeatureScaler,
    MissingImputer,
    RareCategoryGrouper,
    TypeCaster,
)
from mlkit.qc_eda import QualityCheckStep
from mlkit.qc_eda.profile import ProfileOptions
from mlkit.selection import CorrelationFilter, ModelImportanceSelector, VarianceFilter

logger = get_logger(__name__)

SUPPORTED_DATA_SUFFIXES = {".csv", ".parquet", ".pq", ".feather", ".xlsx", ".xls", ".json"}


# --------------------------------------------------------------------------- I/O
def read_table(path: str | Path) -> pd.DataFrame:
    """Read a tabular file by extension (CSV, Parquet, Feather, Excel, JSON lines/records)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if suffix == ".feather":
        return pd.read_feather(p)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    if suffix == ".json":
        try:
            return pd.read_json(p, lines=True)
        except ValueError:
            return pd.read_json(p)
    raise ValueError(
        f"Unsupported data format '{suffix}'. Supported: {sorted(SUPPORTED_DATA_SUFFIXES)}"
    )


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(p, index=False)
    elif suffix == ".json":
        df.to_json(p, orient="records", lines=True)
    else:
        df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------- pipelines
def build_default_pipeline(cfg: TaskConfig) -> Pipeline:
    """Construct the end-to-end 6-stage pipeline.

    Order matters: ``type_caster`` (profile-driven) first converts numbers/booleans/dates
    stored as text, normalises disguised missing values and drops identifiers, so that
    ``datetime`` can expand real dates and ``text`` can summarise free text before imputation
    and encoding. Rare-level folding only touches low-cardinality columns; higher-cardinality
    columns are target/frequency encoded with smoothing instead.
    """
    modelling = cfg.modelling or {}
    profile_options = ProfileOptions(overrides=dict(cfg.column_types)) if cfg.column_types else None
    encoder = CategoricalEncoder(random_state=cfg.seed)
    steps = [
        (
            "qc",
            QualityCheckStep(
                enforce_gate=cfg.quality_gate, profile_options=profile_options, **cfg.qc
            ),
        ),
        ("type_caster", TypeCaster(drop_id=True, profile_options=profile_options)),
        ("datetime", DatetimeFeatures()),
        ("text", TextFeatures(profile_options=profile_options)),
        ("imputer", MissingImputer(add_indicator="auto", categorical_strategy="missing_value")),
        (
            "rare_grouper",
            RareCategoryGrouper(
                min_freq=None, min_count=10, max_cardinality=encoder.low_card_threshold
            ),
        ),
        ("encoder", encoder),
        ("scaler", FeatureScaler()),
        ("variance_filter", VarianceFilter(threshold=0.0)),
        ("corr_filter", CorrelationFilter(threshold=0.95)),
        ("importance_selector", ModelImportanceSelector(top_k=30)),
        (
            "modelling_arena",
            ModelArena(
                cv=cfg.cv_folds,
                enforce_baseline_gate=cfg.baseline_gate,
                random_state=cfg.seed,
                candidates=modelling.get("candidates"),
                handle_imbalance=bool(modelling.get("handle_imbalance", True)),
                calibration=modelling.get("calibration"),
                threshold=modelling.get("threshold", 0.5),
                tuning=modelling.get("tuning"),
            ),
        ),
    ]
    return Pipeline(steps)


def build_pipeline_from_config(cfg: TaskConfig) -> Pipeline:
    """Build a pipeline from ``cfg.steps`` (registry keys) or fall back to the default.

    YAML example::

        steps:
          - {name: qc, step: qc.quality_check}
          - {name: types, step: preprocessing.type_caster}
          - {name: imputer, step: preprocessing.imputer, params: {add_indicator: true}}
          - {name: encoder, step: preprocessing.categorical_encoder}
          - {name: arena, step: modelling.arena, params: {candidates: [linear, lightgbm]}}
    """
    if not cfg.steps:
        return build_default_pipeline(cfg)

    steps: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for i, spec in enumerate(cfg.steps):
        if "step" not in spec:
            raise ValidationError(
                f"steps[{i}] is missing the 'step' key. Available: {list_steps()}"
            )
        name = str(spec.get("name") or spec["step"].split(".")[-1])
        if name in seen:
            raise ValidationError(f"Duplicate step name '{name}' in steps[{i}].")
        seen.add(name)
        params = dict(spec.get("params") or {})
        if spec["step"] == "modelling.arena":
            params.setdefault("cv", cfg.cv_folds)
            params.setdefault("enforce_baseline_gate", cfg.baseline_gate)
            params.setdefault("random_state", cfg.seed)
        steps.append((name, get_step(spec["step"], **params)))

    if not hasattr(steps[-1][1], "predict"):
        raise ValidationError("The last step in 'steps' must be a model (e.g. modelling.arena).")
    return Pipeline(steps)


# ----------------------------------------------------------------- target / split
def _label_key(value: Any) -> str:
    """Canonical text form of a class label (shared by training and evaluation)."""
    return str(value).strip().rstrip(".")


def prepare_target(y: pd.Series, cfg: TaskConfig) -> tuple[pd.Series, dict[str, Any]]:
    """Normalise the target. Returns ``(y, target_info)``.

    - regression: float values;
    - binary: 0/1 (positive = sorted last or ``cfg.modelling['positive_label']``); numeric 0/1
      targets are kept as they are;
    - multiclass: integer codes 0..K-1 in sorted label order (``target_info['label_map']``),
      up to ``cfg.modelling['max_classes']`` classes (default 50).
    """
    info: dict[str, Any] = {"task_type": cfg.task_type}
    if cfg.task_type == "regression":
        y_num = pd.to_numeric(y, errors="coerce")
        if y_num.isna().any():
            raise ValidationError(
                f"Regression target '{cfg.target}' has non-numeric or missing values."
            )
        return y_num.astype(float), info

    if y.isna().any():
        raise ValidationError(
            f"Target '{cfg.target}' contains {int(y.isna().sum())} missing values."
        )
    max_classes = int((cfg.modelling or {}).get("max_classes", 50))
    keys = y.map(_label_key)
    classes = sorted(keys.unique(), key=lambda k: (0, float(k)) if _is_number(k) else (1, k))
    info["classes"] = classes
    info["n_classes"] = len(classes)
    if len(classes) < 2:
        raise ValidationError(f"Target '{cfg.target}' has a single class; nothing to learn.")
    if len(classes) > max_classes:
        raise ValidationError(
            f"Classification target has {len(classes)} classes (> modelling.max_classes={max_classes}). "
            "Group rare classes or use task_type: regression for ordinal/count targets."
        )
    if len(classes) == 2:
        if pd.api.types.is_numeric_dtype(y) and set(pd.unique(y)) <= {0, 1}:
            return y.astype(int), info
        pos = _label_key((cfg.modelling or {}).get("positive_label", classes[1]))
        if pos not in classes:
            raise ValidationError(f"positive_label '{pos}' not found in target classes {classes}.")
        info["positive_label"] = pos
        info["label_map"] = {c: int(c == pos) for c in classes}
        return (keys == pos).astype(int), info
    info["label_map"] = {c: i for i, c in enumerate(classes)}
    return keys.map(info["label_map"]).astype(int), info


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    strategy: str
    fit_params: dict[str, Any] = field(default_factory=dict)


def split_data(
    X: pd.DataFrame, y: pd.Series, cfg: TaskConfig, arena_step: str | None = None
) -> Split:
    """Train/test split respecting ``time_col`` (chronological holdout) and ``group_col``."""
    fit_params: dict[str, Any] = {}

    if cfg.time_col:
        if cfg.time_col not in X.columns:
            raise ValidationError(f"time_col '{cfg.time_col}' not found in data.")
        t = pd.to_datetime(X[cfg.time_col], errors="coerce")
        if t.isna().all():
            t = pd.to_numeric(X[cfg.time_col], errors="coerce")
        order = np.argsort(t.to_numpy(), kind="stable")
        Xs, ys = X.iloc[order], y.iloc[order]
        cut = int(round(len(Xs) * (1 - cfg.test_size)))
        X_train, X_test, y_train, y_test = (
            Xs.iloc[:cut],
            Xs.iloc[cut:],
            ys.iloc[:cut],
            ys.iloc[cut:],
        )
        if arena_step:
            fit_params[f"{arena_step}__time_index"] = t.iloc[order].iloc[:cut].to_numpy()
        strategy = "time"
    elif cfg.group_col:
        if cfg.group_col not in X.columns:
            raise ValidationError(f"group_col '{cfg.group_col}' not found in data.")
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed)
        tr, te = next(gss.split(X, y, groups=X[cfg.group_col]))
        X_train, X_test, y_train, y_test = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        if arena_step:
            fit_params[f"{arena_step}__groups"] = X_train[cfg.group_col].to_numpy()
        strategy = "group"
    else:
        stratify = y if cfg.task_type == "classification" and cfg.stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=stratify
        )
        strategy = "stratified" if stratify is not None else "random"

    return Split(X_train, X_test, y_train, y_test, strategy, fit_params)


def _threshold_failures(cfg: TaskConfig, values: dict[str, Any]) -> list[str]:
    """Human-readable list of ``metric_thresholds`` violated by ``values``."""
    failures = []
    for metric, threshold in cfg.metric_thresholds.items():
        value = values.get(metric)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            failures.append(f"{metric} could not be computed")
            continue
        if metric in LOWER_IS_BETTER:
            if value > threshold:
                failures.append(f"{metric}={value:.4f} (must be <= {threshold})")
        elif value < threshold:
            failures.append(f"{metric}={value:.4f} (must be >= {threshold})")
    return failures


def validate_metric_thresholds(cfg: TaskConfig) -> None:
    """Fail fast on unknown metric names so a typo cannot silently disable the release gate."""
    allowed = AVAILABLE_METRICS[cfg.task_type]
    unknown = sorted(set(cfg.metric_thresholds) - allowed)
    if unknown:
        raise ValidationError(
            f"Unknown metric(s) in metric_thresholds for {cfg.task_type}: {unknown}. "
            f"Available: {sorted(allowed)}"
        )


def _scores(
    pipeline: Pipeline, X: pd.DataFrame, task_type: str
) -> tuple[np.ndarray, np.ndarray | None]:
    preds = np.asarray(pipeline.predict(X))
    probs = None
    if task_type == "classification":
        try:
            probs = np.asarray(pipeline.predict_proba(X))
        except (AttributeError, NotImplementedError):
            probs = None
    return preds, probs


def _arena_step_name(pipeline: Pipeline) -> str | None:
    for name, step in pipeline.steps:
        if isinstance(step, ModelArena):
            return name
    return None


# ------------------------------------------------------------------------ train
@dataclass
class TrainResult:
    pipeline: Pipeline
    cfg: TaskConfig
    metrics: dict[str, Any]
    monitor: DriftMonitor
    drift_report: Report
    bundle_dir: Path | None = None


def train(
    cfg: TaskConfig,
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
    html_report: str | Path | None = None,
) -> TrainResult:
    """Train, evaluate on a holdout, capture drift baselines and (optionally) save a bundle."""
    if cfg.target not in df.columns:
        raise ValidationError(
            f"Target column '{cfg.target}' not found in dataset columns: {list(df.columns)}"
        )
    validate_metric_thresholds(cfg)

    X = df.drop(columns=[cfg.target])
    y, target_info = prepare_target(df[cfg.target], cfg)

    pipeline = build_pipeline_from_config(cfg)
    arena_name = _arena_step_name(pipeline)
    split = split_data(X, y, cfg, arena_step=arena_name)
    logger.info(
        "Split data: %d train / %d test rows (strategy=%s)",
        len(split.X_train),
        len(split.X_test),
        split.strategy,
    )

    logger.info("Training %d-step pipeline (task=%s)", len(pipeline.steps), cfg.task_type)
    pipeline.fit(split.X_train, split.y_train, cfg=cfg, **split.fit_params)

    metrics: dict[str, Any] = {"split": split.strategy, "target": target_info}
    arena = pipeline.named_steps.get(arena_name) if arena_name else None
    if arena is not None:
        metrics["champion"] = arena.best_model_name_
        metrics["cv_scope"] = arena.cv_scope_
        metrics["calibration"] = arena.calibration_
        metrics["decision_threshold"] = arena.decision_threshold_
        metrics["tuned_params"] = getattr(arena, "tuned_params_", {})
        metrics["cv"] = {
            k: v
            for k, v in arena.cv_results_.get(arena.best_model_name_, {}).items()
            if k != "folds"
        }
        metrics["arena"] = arena.comparison_table_.to_dict(orient="records")
        logger.info("Champion model: %s", arena.best_model_name_)

    preds, probs = _scores(pipeline, split.X_test, cfg.task_type)
    metrics["holdout"] = compute_metrics(
        np.asarray(split.y_test), preds, probs, task_type=cfg.task_type
    )
    metrics["holdout_ci"] = compute_bootstrap_ci(
        np.asarray(split.y_test), preds, probs, task_type=cfg.task_type, random_state=cfg.seed
    )
    logger.info(
        "Holdout metrics: %s", json.dumps({k: round(v, 4) for k, v in metrics["holdout"].items()})
    )

    # Thresholds from config act as a release gate on the holdout set
    failures = _threshold_failures(cfg, metrics["holdout"])
    metrics["thresholds_passed"] = not failures
    metrics["threshold_failures"] = failures
    if failures:
        logger.warning("Metric thresholds not met: %s", "; ".join(failures))

    monitor = DriftMonitor(**cfg.monitoring)
    # Identifier / time / group keys are expected to change between batches: not drift features.
    key_cols = [
        c for c in (cfg.id_col, cfg.time_col, cfg.group_col) if c and c in split.X_train.columns
    ]
    monitor.fit(split.X_train.drop(columns=key_cols), split.y_train, cfg=cfg)
    train_scores = model_output(pipeline, split.X_train, cfg.task_type)
    monitor.fit_predictions(train_scores)
    drift_report = monitor.monitor(
        split.X_test, y_pred=model_output(pipeline, split.X_test, cfg.task_type)
    )

    result = TrainResult(pipeline, cfg, metrics, monitor, drift_report)

    if output_dir is not None:
        result.bundle_dir = pipeline.save(
            path=output_dir,
            cfg=cfg,
            X_sample=split.X_train.head(100),
            baseline=monitor.to_dict(),
            metrics=metrics,
        )
        logger.info("Bundle saved to %s", result.bundle_dir)

    if html_report is not None:
        pipeline.to_html_report(
            output_path=html_report,
            summary_stats={
                "Champion": metrics.get("champion", "-"),
                **{f"Holdout {k}": f"{v:.4f}" for k, v in metrics["holdout"].items()},
            },
        )
        logger.info("HTML report written to %s", html_report)

    return result


def model_output(pipeline: Pipeline, X: pd.DataFrame, task_type: str) -> np.ndarray:
    """Scalar model output used for prediction-drift: P(class=1) (binary), confidence of the
    predicted class (multiclass) or the regression value."""
    preds, probs = _scores(pipeline, X, task_type)
    if probs is not None and probs.ndim == 2 and probs.shape[1] > 2:
        return probs.max(axis=1)
    if probs is not None and probs.ndim == 2 and probs.shape[1] == 2:
        return probs[:, 1]
    return preds.astype(float)


# ---------------------------------------------------------------------- predict
def validate_input(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    """Check that all training columns exist; drop unknown extra columns, keep training order."""
    if not schema:
        return df
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise ValidationError(f"Input is missing {len(missing)} required column(s): {missing}")
    extra = [c for c in df.columns if c not in schema]
    if extra:
        logger.warning("Ignoring %d unknown column(s): %s", len(extra), extra[:10])
    return df[list(schema)]


def predict_frame(
    pipeline: Pipeline,
    cfg: TaskConfig,
    metadata: dict[str, Any],
    df: pd.DataFrame,
    threshold: float | None = None,
    explain: int = 0,
) -> pd.DataFrame:
    """Score a DataFrame with a loaded bundle and return a predictions frame.

    ``threshold=None`` uses the decision threshold stored in the bundle (tuned or configured
    at training time, default 0.5).
    """
    if threshold is None:
        threshold = float(metadata.get("metrics", {}).get("decision_threshold") or 0.5)
    X = df.drop(columns=[cfg.target], errors="ignore")
    X = validate_input(X, metadata.get("schema", {}))

    out = pd.DataFrame(index=df.index)
    if cfg.id_col and cfg.id_col in df.columns:
        out[cfg.id_col] = df[cfg.id_col].to_numpy()

    if cfg.task_type == "classification":
        label_map = (metadata.get("metrics", {}).get("target") or {}).get("label_map")
        inverse = {v: k for k, v in (label_map or {}).items()}
        try:
            proba = np.asarray(pipeline.predict_proba(X))
            if proba.ndim == 2 and proba.shape[1] > 2:  # multiclass: one score per class
                for k in range(proba.shape[1]):
                    out[f"score_{inverse.get(k, k)}"] = proba[:, k]
                out["score"] = proba.max(axis=1)  # confidence of the predicted class
                out["prediction"] = proba.argmax(axis=1)
            else:
                score = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()
                out["score"] = score
                out["prediction"] = (score >= threshold).astype(int)
        except AttributeError:
            out["prediction"] = pipeline.predict(X)
        if inverse:
            out["label"] = out["prediction"].map(inverse)
    else:
        out["prediction"] = pipeline.predict(X)
    out = out.reset_index(drop=True)
    if explain:
        from mlkit.explain import explain_frame

        reasons = explain_frame(pipeline, X, task_type=cfg.task_type, top_k=int(explain))
        out = pd.concat([out, reasons], axis=1)
    return out


def predict_bundle(
    bundle_path: str | Path, df: pd.DataFrame, threshold: float | None = None, explain: int = 0
) -> pd.DataFrame:
    pipeline, cfg, metadata = load_bundle(bundle_path)
    return predict_frame(pipeline, cfg, metadata, df, threshold=threshold, explain=explain)


# ---------------------------------------------------------------------- monitor
def monitor_bundle(bundle_path: str | Path, df: pd.DataFrame) -> Report:
    """Compare a new batch against the baseline stored in the bundle (data + prediction drift)."""
    pipeline, cfg, metadata = load_bundle(bundle_path)
    baseline = metadata.get("baseline") or {}
    if not baseline:
        raise ValidationError("Bundle has no drift baseline (baseline.json is empty).")
    monitor = DriftMonitor.from_dict(baseline)

    X = df.drop(columns=[cfg.target], errors="ignore")
    y_pred = None
    try:
        X_valid = validate_input(X, metadata.get("schema", {}))
        y_pred = model_output(pipeline, X_valid, cfg.task_type)
    except ValidationError as exc:
        logger.warning("Skipping prediction drift: %s", exc)
    return monitor.monitor(X, y_pred=y_pred)


# --------------------------------------------------------------------- evaluate
@dataclass
class EvaluationResult:
    overall: dict[str, Any]
    by_group: pd.DataFrame | None
    failures: list[str]
    reference: dict[str, float]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "by_group": None if self.by_group is None else self.by_group.to_dict(orient="records"),
            "failures": self.failures,
            "reference_holdout": self.reference,
            "passed": self.passed,
        }


def encode_target_like_training(
    y: pd.Series, cfg: TaskConfig, metadata: dict[str, Any]
) -> np.ndarray:
    """Encode labels exactly as during training (stored ``label_map``), never re-deriving it."""
    if cfg.task_type == "regression":
        return pd.to_numeric(y, errors="raise").to_numpy(dtype=float)
    label_map = (metadata.get("metrics", {}).get("target") or {}).get("label_map")
    if not label_map:
        return y.to_numpy()
    cleaned = y.astype(str).str.strip().str.rstrip(".")
    unknown = sorted(set(cleaned) - set(label_map))
    if unknown:
        raise ValidationError(f"Labels not seen at training time: {unknown[:10]}")
    return cleaned.map(label_map).to_numpy(dtype=int)


def _labelled_metrics(
    y: np.ndarray, pred: np.ndarray, probs: np.ndarray | None, task_type: str
) -> dict[str, Any]:
    m: dict[str, Any] = {"n": int(len(y))}
    binary = probs is None or probs.ndim == 1 or probs.shape[1] <= 2
    if task_type == "classification" and binary:
        m["positive_rate"] = float(np.mean(y))
        if probs is not None:
            m["mean_score"] = float(np.mean(probs[:, 1] if probs.ndim == 2 else probs))
    elif task_type == "classification":
        m["mean_confidence"] = float(np.mean(probs.max(axis=1)))  # type: ignore[union-attr]
    m.update(compute_metrics(y, pred, probs, task_type=task_type))  # type: ignore[arg-type]
    if task_type == "classification" and len(np.unique(y)) < 2:
        m["roc_auc"] = float("nan")  # undefined on a single-class batch (not 0.5)
    return m


def evaluate_frame(
    pipeline: Pipeline,
    cfg: TaskConfig,
    metadata: dict[str, Any],
    df: pd.DataFrame,
    by: str | None = None,
    threshold: float | None = None,
) -> EvaluationResult:
    """Score labelled data and compute metrics overall and per ``by`` group (e.g. cutoff/week).

    Each group (and the overall set) is checked against ``cfg.metric_thresholds``.
    """
    if cfg.target not in df.columns:
        raise ValidationError(f"Evaluation data needs the target column '{cfg.target}'.")
    if by is not None and by not in df.columns:
        raise ValidationError(f"Group column '{by}' not found in evaluation data.")

    y = encode_target_like_training(df[cfg.target], cfg, metadata)
    preds = predict_frame(pipeline, cfg, metadata, df, threshold=threshold)
    pred = preds["prediction"].to_numpy()
    probs = None
    if cfg.task_type == "classification" and "score" in preds.columns:
        score_cols = [c for c in preds.columns if c.startswith("score_")]
        if score_cols:  # multiclass: columns are in class-code order
            probs = preds[score_cols].to_numpy(dtype=float)
        else:
            s1 = preds["score"].to_numpy(dtype=float)
            probs = np.column_stack([1 - s1, s1])

    overall = _labelled_metrics(y, pred, probs, cfg.task_type)
    failures = [f"overall: {f}" for f in _threshold_failures(cfg, overall)]

    by_group = None
    if by is not None:
        rows = []
        keys = df[by].to_numpy()
        for key in pd.unique(keys):
            mask = keys == key
            m = _labelled_metrics(
                y[mask], pred[mask], None if probs is None else probs[mask], cfg.task_type
            )
            rows.append({by: key, **m})
            failures += [f"{by}={key}: {f}" for f in _threshold_failures(cfg, m)]
        by_group = pd.DataFrame(rows).sort_values(by).reset_index(drop=True)

    reference = {
        k: v
        for k, v in (metadata.get("metrics", {}).get("holdout") or {}).items()
        if isinstance(v, (int, float))
    }
    return EvaluationResult(overall, by_group, failures, reference)


def evaluate_bundle(
    bundle_path: str | Path, df: pd.DataFrame, by: str | None = None, threshold: float | None = None
) -> EvaluationResult:
    pipeline, cfg, metadata = load_bundle(bundle_path)
    return evaluate_frame(pipeline, cfg, metadata, df, by=by, threshold=threshold)
