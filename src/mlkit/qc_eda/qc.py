"""Quality Control checks with gatekeeper enforcement, built on :mod:`mlkit.qc_eda.profile`.

Every check reads the shared :class:`~mlkit.qc_eda.profile.DatasetProfile`, so QC, EDA and
preprocessing agree on what each column *is*. Column-level findings are also collected in
``issues_`` (column, check, severity, detail, suggestion) for reports and dashboards.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mlkit.core.config import TaskConfig
from mlkit.core.registry import register
from mlkit.core.report import Severity
from mlkit.core.step import BaseStep
from mlkit.qc_eda.profile import DatasetProfile, ProfileOptions, apply_coercion, profile_dataframe


def univariate_signal(
    x: pd.Series, y: pd.Series, kind: str, min_rows_per_bin: int = 5
) -> float | None:
    """How completely a single feature determines the target, in [0, 1] (None if not assessable).

    The feature is bucketed (quantile bins for numbers, its own values for categories) with at
    least ``min_rows_per_bin`` rows per bucket. A score near 1 requires (almost) *every* bucket to be
    pure, which is vanishingly unlikely by chance even for small buckets; at least 30 rows are needed.

    - binary target + numeric feature: rank separation ``2 * |AUC - 0.5|``;
    - other classification: normalised purity ``(acc - base) / (1 - base)`` where ``acc`` is the share
      of rows matching their bucket's majority class and ``base`` the global majority share.
    - regression: correlation ratio squared (share of target variance explained by buckets).
    """
    mask = (x.notna() & y.notna()).to_numpy()
    n = int(mask.sum())
    max_bins = n // min_rows_per_bin
    if n < 30 or max_bins < 2:
        return None
    xv = x.to_numpy()[mask]
    if pd.api.types.is_numeric_dtype(x):
        xv = xv.astype(float)
        if len(np.unique(xv)) > max_bins:
            q = min(20, max_bins)
            ranks = np.argsort(np.argsort(xv, kind="stable"), kind="stable")
            codes = (ranks * q // n).astype(np.int64)
        else:
            codes = pd.factorize(xv)[0]
    else:
        codes = pd.factorize(pd.Series(xv).map(str).to_numpy())[0]  # map(str): lists/dicts too
        if codes.max() + 1 > max_bins:
            return None  # too many categories for a trustworthy estimate (identifier-like)
    n_bins = int(codes.max()) + 1
    if n_bins < 2:
        return None
    yv = y.to_numpy()[mask]

    if kind == "regression":
        yf = pd.to_numeric(pd.Series(yv), errors="coerce").to_numpy(dtype=float)
        total = float(((yf - yf.mean()) ** 2).sum())
        if total == 0:
            return None
        cnt = np.bincount(codes, minlength=n_bins)
        means = np.bincount(codes, weights=yf, minlength=n_bins) / np.maximum(cnt, 1)
        return float(1 - ((yf - means[codes]) ** 2).sum() / total)

    y_codes, y_uniques = pd.factorize(yv.astype(str))
    n_cls = len(y_uniques)
    if n_cls < 2:
        return None
    if n_cls == 2 and pd.api.types.is_numeric_dtype(x):
        # Binary target + numeric feature: rank separation 2*|AUC - 0.5| (1.0 = a threshold on this
        # feature splits the classes perfectly, wherever that threshold lies).
        from scipy.stats import rankdata

        ranks = rankdata(x.to_numpy()[mask].astype(float))
        pos = y_codes == 1
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return float(2 * abs(auc - 0.5))
    table = np.bincount(codes * n_cls + y_codes, minlength=n_bins * n_cls).reshape(n_bins, n_cls)
    base = float(np.bincount(y_codes).max() / n)
    acc = float(table.max(axis=1).sum() / n)
    return max(0.0, (acc - base) / (1 - base))


@register("qc.quality_check")
class QualityCheckStep(BaseStep):
    """Automated data quality inspection on top of the dataset profile.

    The step never changes the data (``transform`` is the identity); it reports, scores and,
    with ``enforce_gate=True``, stops the pipeline on errors.
    """

    #: Inspection only: pipeline-level CV does not need to re-run it inside every fold.
    fold_passthrough = True

    def __init__(
        self,
        max_missing_rate: float = 0.5,
        max_cardinality: int = 100,
        leakage_threshold: float = 0.99,
        enforce_gate: bool = True,
        min_rows: int = 50,
        duplicate_rate_warn: float = 0.05,
        quasi_constant_share: float = 0.99,
        outlier_rate_warn: float = 0.05,
        imbalance_warn: float = 0.05,
        leakage_warn_threshold: float = 0.97,
        missing_leakage_warn: float = 0.8,
        leakage_warn_threshold_regression: float = 0.9,
        profile_options: ProfileOptions | None = None,
    ) -> None:
        super().__init__()
        self.max_missing_rate = max_missing_rate
        self.max_cardinality = max_cardinality
        self.leakage_threshold = leakage_threshold
        self.enforce_gate = enforce_gate
        self.min_rows = min_rows
        self.duplicate_rate_warn = duplicate_rate_warn
        self.quasi_constant_share = quasi_constant_share
        self.outlier_rate_warn = outlier_rate_warn
        self.imbalance_warn = imbalance_warn
        self.leakage_warn_threshold = leakage_warn_threshold
        self.missing_leakage_warn = missing_leakage_warn
        self.leakage_warn_threshold_regression = leakage_warn_threshold_regression
        self.profile_options = profile_options
        self.data_quality_score_: float = 100.0

    # ----------------------------------------------------------------- helpers
    def _issue(
        self, column: str, check: str, severity: Severity, detail: str, suggestion: str = ""
    ) -> None:
        self.issues_.append(
            {
                "column": column,
                "check": check,
                "severity": severity,
                "detail": detail,
                "suggestion": suggestion,
            }
        )

    def _report_columns(
        self,
        name: str,
        cols: list[str],
        severity: Severity,
        what: str,
        ok_msg: str,
        suggestion: str = "",
    ) -> None:
        if cols:
            shown = cols[:8] + (["..."] if len(cols) > 8 else [])
            self.report_.add(
                name,
                severity,
                f"{len(cols)} {what}: {shown}",
                {"columns": cols, "suggestion": suggestion},
            )
        else:
            self.report_.add(name, "ok", ok_msg)

    # --------------------------------------------------------------------- fit
    def _fit(
        self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None
    ) -> QualityCheckStep:
        self.report_.title = "Quality Control & EDA Report"
        self.issues_: list[dict[str, Any]] = []

        if X.empty or len(X.columns) == 0:
            self.report_.add("empty_data", "error", "Input DataFrame is completely empty.")
            self.data_quality_score_ = 0.0
            self.profile_: DatasetProfile = profile_dataframe(X)
            if self.enforce_gate and (cfg is None or cfg.quality_gate):
                self.report_.raise_if_failed()
            return self

        task_type = cfg.task_type if cfg else None
        y_s = None if y is None else pd.Series(np.asarray(y), name=cfg.target if cfg else "target")
        prof = profile_dataframe(X, y=y_s, task_type=task_type, options=self.profile_options)
        self.profile_ = prof
        cols = prof.columns

        # 1. Shape
        self.report_.add(
            "data_shape",
            "ok",
            f"Dataset contains {prof.n_rows} rows and {prof.n_cols} columns ({prof.memory_mb:.1f} MB).",
            {"rows": prof.n_rows, "cols": prof.n_cols},
        )
        if prof.n_rows < self.min_rows:
            self.report_.add(
                "too_few_rows",
                "warn",
                f"Only {prof.n_rows} rows (< {self.min_rows}): estimates will be unstable.",
            )

        # 2. Column names
        names = [c for c in X.columns]
        dup_names = sorted({str(c) for c in names if names.count(c) > 1})
        odd = [str(c) for c in names if not isinstance(c, str) or str(c) != str(c).strip()]
        if dup_names or odd:
            self.report_.add(
                "column_names",
                "warn",
                f"Duplicate names: {dup_names[:5]}; non-string or padded names: {odd[:5]}",
                {"duplicates": dup_names, "irregular": odd},
            )

        # 3. Duplicate rows / columns
        dup_pct = prof.n_duplicate_rows / prof.n_rows
        if dup_pct > self.duplicate_rate_warn:
            self.report_.add(
                "duplicates",
                "warn",
                f"{prof.n_duplicate_rows} duplicate rows ({dup_pct * 100:.1f}%) detected.",
            )
        else:
            self.report_.add(
                "duplicates",
                "ok",
                f"Duplicate rows check passed ({prof.n_duplicate_rows} duplicates).",
            )
        if prof.duplicate_columns:
            self.report_.add(
                "duplicate_columns",
                "warn",
                f"{len(prof.duplicate_columns)} groups of identical columns: {prof.duplicate_columns[:3]}",
                {"groups": prof.duplicate_columns},
            )
            for group in prof.duplicate_columns:
                for c in group[1:]:
                    self._issue(
                        c,
                        "duplicate_columns",
                        "warn",
                        f"identical to '{group[0]}'",
                        "drop the copy",
                    )

        # 4. Missing values (incl. disguised tokens such as 'n/a')
        high_missing = [
            c
            for c, p in cols.items()
            if p.missing_rate > self.max_missing_rate and p.semantic_type != "empty"
        ]
        empty = prof.columns_of("empty")
        self._report_columns(
            "missing_values",
            high_missing,
            "warn",
            f"columns exceed missing threshold ({self.max_missing_rate * 100:.0f}%)",
            "Missing values within acceptable thresholds.",
            "impute with indicators, or drop if uninformative",
        )
        for c in high_missing:
            self._issue(
                c,
                "missing_values",
                "warn",
                f"{cols[c].missing_rate:.0%} missing",
                "impute + missing indicator",
            )
        if empty:
            self._report_columns(
                "empty_columns", empty, "warn", "columns are entirely missing", "", "drop"
            )
        disguised = [c for c, p in cols.items() if any("disguised missing" in n for n in p.notes)]
        if disguised:
            self._report_columns(
                "disguised_missing",
                disguised,
                "warn",
                "columns contain missing-value tokens ('n/a', 'null', '-')",
                "",
                "treated as missing by TypeCaster",
            )

        # 5. Constant / quasi-constant
        constant = prof.columns_of("constant")
        self._report_columns(
            "constant_columns",
            constant,
            "warn",
            "constant columns detected",
            "No constant columns detected.",
            "drop",
        )
        quasi = [
            c
            for c, p in cols.items()
            if p.semantic_type not in ("constant", "empty")
            and p.dominant_share >= self.quasi_constant_share
        ]
        if quasi:
            self._report_columns(
                "quasi_constant",
                quasi,
                "warn",
                f"columns have one value in >= {self.quasi_constant_share:.0%} of rows",
                "",
                "usually uninformative; check before keeping",
            )

        # 6. Identifiers and free text
        ids = prof.columns_of("identifier")
        if ids:
            self._report_columns(
                "identifier_columns",
                ids,
                "warn",
                "identifier-like columns (unique per row)",
                "",
                "excluded from features by TypeCaster",
            )
            for c in ids:
                self._issue(
                    c,
                    "identifier_columns",
                    "warn",
                    f"{cols[c].unique_ratio:.0%} unique",
                    "exclude from features",
                )
        text = prof.columns_of("text")
        if text:
            self._report_columns(
                "text_columns",
                text,
                "warn",
                "free-text columns",
                "",
                "use TextFeatures (length/word statistics) or a dedicated NLP pipeline",
            )

        # 7. Cardinality
        cat_like = [
            c for c, p in cols.items() if p.semantic_type in ("categorical", "high_cardinality")
        ]
        high_card = [c for c in cat_like if cols[c].n_unique > self.max_cardinality]
        self._report_columns(
            "high_cardinality",
            high_card,
            "warn",
            f"categorical columns have cardinality > {self.max_cardinality}",
            "All categorical features have healthy cardinality.",
            "target / frequency encoding (never one-hot)",
        )

        # 8. Types stored as text, mixed types, infinities, outliers
        coerced = {c: p.coercion for c, p in cols.items() if p.coercion}
        if coerced:
            self.report_.add(
                "types_as_text",
                "ok",
                f"{len(coerced)} columns stored as text will be converted: {coerced}",
                {"columns": coerced},
            )
        if prof.duplicate_names:
            self._report_columns(
                "duplicate_names",
                prof.duplicate_names,
                "warn",
                "column names appear more than once",
                "",
                "only the first occurrence is used; rename upstream",
            )
        unsupported = prof.columns_of("unsupported")
        if unsupported:
            self._report_columns(
                "unsupported_columns",
                unsupported,
                "warn",
                "columns could not be analysed (see profile notes)",
                "",
                "excluded from features; convert them upstream or set column_types",
            )
            for c in unsupported:
                self._issue(
                    c, "unsupported_columns", "warn", "; ".join(cols[c].notes), "convert upstream"
                )
        ambiguous = [c for c, p in cols.items() if p.ambiguous]
        if ambiguous:
            self._report_columns(
                "ambiguous_types",
                ambiguous,
                "warn",
                "columns whose type is a judgement call",
                "",
                "confirm with TaskConfig.column_types (e.g. {'stars': 'categorical'})",
            )
            for c in ambiguous:
                note = next(n for n in cols[c].notes if n.startswith("ambiguous"))
                self._issue(c, "ambiguous_types", "warn", note, "set column_types")
        mixed = [c for c, p in cols.items() if any(n.startswith("mixed types") for n in p.notes)]
        if mixed:
            self._report_columns(
                "mixed_types",
                mixed,
                "warn",
                "columns mix numbers and text",
                "",
                "fix upstream or map the non-numeric tokens",
            )
        infinite = [c for c, p in cols.items() if p.numeric.get("n_inf", 0) > 0]
        if infinite:
            self._report_columns(
                "infinite_values",
                infinite,
                "warn",
                "columns contain +/-inf",
                "",
                "treated as missing by TypeCaster",
            )
        outliers = [
            c
            for c, p in cols.items()
            if p.semantic_type == "numeric"
            and p.numeric.get("outlier_rate", 0) > self.outlier_rate_warn
        ]
        if outliers:
            self._report_columns(
                "outliers",
                outliers,
                "warn",
                f"numeric columns with > {self.outlier_rate_warn:.0%} extreme values (3 x IQR)",
                "",
                "consider OutlierClipper / log transform; tree models are robust",
            )

        # 9. Target
        if y_s is not None:
            self._check_target(prof, cfg)
            self._check_leakage(X, y_s, prof)

        # 10. Overall Data Quality Score (0 - 100)
        self._score(X, prof, dup_pct, constant)

        if self.enforce_gate and (cfg is None or cfg.quality_gate) and self.report_.has_errors:
            self.report_.raise_if_failed(gate_name="QC Quality Gate")
        return self

    def _check_target(self, prof: DatasetProfile, cfg: TaskConfig | None) -> None:
        t = prof.target
        assert t is not None
        if t.kind == "empty":
            self.report_.add(
                "target_distribution", "error", "Target variable is completely empty or null."
            )
            return
        if t.kind == "constant":
            self.report_.add(
                "target_distribution",
                "error",
                "Target contains only 1 unique class. Cannot train a supervised model.",
            )
            return
        if t.n_missing:
            self.report_.add(
                "target_missing",
                "warn",
                f"Target has {t.n_missing} missing values (rows will be unusable).",
            )
        if t.kind == "regression":
            skew = t.numeric.get("skew", 0.0)
            msg = f"Regression target: mean={t.numeric.get('mean', float('nan')):.4g}, skew={skew:.2f}."
            self.report_.add(
                "target_distribution",
                "warn" if abs(skew) > 2 else "ok",
                msg + (" Strong skew: consider a log transform." if abs(skew) > 2 else ""),
            )
            return
        n_classes = len(t.class_counts)
        self.report_.add(
            "target_distribution",
            "ok",
            f"Target variable has {n_classes} unique values: {dict(list(t.class_counts.items())[:10])}",
            {"class_counts": t.class_counts},
        )
        folds = cfg.cv_folds if cfg else 5
        smallest = min(t.class_counts.values())
        if smallest < folds:
            self.report_.add(
                "rare_classes",
                "error",
                f"Smallest class has {smallest} rows (< cv_folds={folds}): stratified CV impossible.",
            )
        elif t.minority_share is not None and t.minority_share < self.imbalance_warn:
            self.report_.add(
                "class_imbalance",
                "warn",
                f"Minority class is {t.minority_share:.2%} of rows: prefer PR-AUC / calibrated probabilities "
                "and avoid class weights when probabilities matter.",
                {"minority_share": t.minority_share},
            )

    def _check_leakage(self, X: pd.DataFrame, y: pd.Series, prof: DatasetProfile) -> None:
        t = prof.target
        if t is None or t.kind in ("empty", "constant"):
            return
        kind = "regression" if t.kind == "regression" else "classification"
        y_r = y.reset_index(drop=True)
        by_name = {str(c): c for c in X.columns}
        suspects: list[tuple[str, float]] = []
        warn_values: list[tuple[str, float]] = []
        warn_missing: list[tuple[str, float]] = []
        for c, p in prof.columns.items():
            if p.semantic_type in (
                "empty",
                "constant",
                "identifier",
                "text",
                "ignored",
                "unsupported",
            ):
                continue
            col = X[by_name[c]]
            if isinstance(col, pd.DataFrame):  # duplicated column names
                col = col.iloc[:, 0]
            try:
                x = apply_coercion(col.reset_index(drop=True), p.semantic_type, p.coercion)
                if pd.api.types.is_datetime64_any_dtype(x):
                    x = pd.Series(
                        x.to_numpy(dtype="datetime64[ns]").astype("int64"), dtype=float
                    ).where(x.notna())
            except Exception:  # noqa: BLE001 - leakage screening must not crash on odd columns
                continue
            try:
                if x.astype(str).equals(y_r.astype(str)):
                    suspects.append((c, 1.0))
                    continue
                score = univariate_signal(x, y_r, kind)
                if kind == "regression" and pd.api.types.is_numeric_dtype(x):
                    valid = x.notna() & y_r.notna()
                    if valid.sum() > 20:
                        rho = (
                            x[valid].rank().corr(pd.to_numeric(y_r[valid], errors="coerce").rank())
                        )
                        if not np.isnan(rho):
                            score = max(score or 0.0, abs(float(rho)))
            except Exception:  # noqa: BLE001 - one odd column must not stop QC
                continue
            if score is not None and score >= self.leakage_threshold:
                suspects.append((c, round(score, 4)))
                continue
            # |Spearman| / eta^2 (regression) and rank separation (classification) live on
            # different scales; thresholds were calibrated on real datasets (see ADR-008).
            warn_at = (
                self.leakage_warn_threshold_regression
                if kind == "regression"
                else self.leakage_warn_threshold
            )
            if score is not None and score >= warn_at:
                warn_values.append((c, round(score, 4)))
            # Leakage through *missingness*: fields filled in only after the outcome (lifeboat
            # number, cancellation date...) are missing for one class and present for the other.
            if 0 < p.missing_rate < 1:
                try:
                    miss = univariate_signal(
                        col.reset_index(drop=True).isna().astype(int), y_r, kind
                    )
                except Exception:  # noqa: BLE001
                    miss = None
                if miss is not None and miss >= self.missing_leakage_warn:
                    warn_missing.append((c, round(miss, 4)))
        if warn_values or warn_missing:
            parts = []
            if warn_values:
                parts.append(f"near-deterministic values {warn_values}")
            if warn_missing:
                parts.append(
                    f"whether the value is missing almost determines the target {warn_missing}"
                )
            self.report_.add(
                "leakage_suspect",
                "warn",
                "Possible target leakage: "
                + "; ".join(parts)
                + ". Typical of fields recorded after the outcome - verify they exist at prediction time.",
                {"value_signal": warn_values, "missing_signal": warn_missing},
            )
            for c, sc in warn_values + warn_missing:
                self._issue(
                    c,
                    "leakage_suspect",
                    "warn",
                    f"single-feature signal {sc}",
                    "confirm it is available before the prediction time",
                )
        if suspects:
            self.report_.add(
                "data_leakage",
                "error",
                f"Suspicious target leakage detected (single-feature signal >= {self.leakage_threshold}): {suspects}",
                {"leakage_columns": suspects},
            )
            for c, sc in suspects:
                self._issue(
                    c,
                    "data_leakage",
                    "error",
                    f"determines the target alone (signal={sc})",
                    "verify it is known before the prediction time; otherwise drop",
                )
        else:
            self.report_.add("data_leakage", "ok", "No obvious target leakage detected.")

    def _score(
        self, X: pd.DataFrame, prof: DatasetProfile, dup_pct: float, constant: list[str]
    ) -> None:
        score = 100.0
        mean_missing = (
            float(np.mean([p.missing_rate for p in prof.columns.values()])) if prof.columns else 0.0
        )
        score -= min(35.0, mean_missing * 70.0)
        score -= min(20.0, dup_pct * 100.0)
        if prof.n_cols:
            score -= min(20.0, (len(constant) + len(prof.columns_of("empty"))) / prof.n_cols * 50.0)
        if self.report_.has_errors:
            score -= 25.0
        self.data_quality_score_ = float(max(0.0, min(100.0, round(score, 1))))
        status: Severity = (
            "ok"
            if self.data_quality_score_ >= 75
            else ("warn" if self.data_quality_score_ >= 50 else "error")
        )
        self.report_.add(
            "data_quality_score",
            status,
            f"Overall Data Quality Score: {self.data_quality_score_} / 100",
            {"score": self.data_quality_score_},
        )

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X  # inspection only

    @property
    def issues_frame_(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.issues_, columns=["column", "check", "severity", "detail", "suggestion"]
        )


def run_qc(
    df: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None, **params: Any
) -> QualityCheckStep:
    """Convenience runner for QC inspection. Pass ``enforce_gate=False`` to inspect without raising."""
    params.setdefault("enforce_gate", True)
    step = QualityCheckStep(**params)
    step.fit(df, y=y, cfg=cfg)
    return step
