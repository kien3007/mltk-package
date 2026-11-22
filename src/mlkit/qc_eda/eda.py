"""Exploratory Data Analysis that works for any column type and any target type.

``run_eda`` combines the dataset profile, the QC findings and type-aware statistics:

- target association per feature on a common [0, 1] scale, with the method chosen from the
  (feature type, target type) pair: |Spearman| (numeric~numeric), correlation ratio eta
  (numeric~categorical), bias-corrected Cramer's V (categorical~categorical);
- a feature-feature association matrix with the same measures (top features only);
- target rate / mean per numeric bin or category (top categories + "other");
- missingness vs. target (does "being missing" carry signal?).

Binary, multiclass and regression targets are supported, as is having no target at all:
EDA is not restricted by what the modelling stage supports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlkit.qc_eda.profile import (
    DatasetProfile,
    ProfileOptions,
    apply_coercion,
    deduplicate_columns,
)

_SKIP = ("empty", "constant", "identifier", "ignored", "unsupported")


# ------------------------------------------------------------------ association measures
def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramer's V between two categorical series, in [0, 1]."""
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return float("nan")
    xc = pd.factorize(x[mask].astype(str))[0]
    yc = pd.factorize(y[mask].astype(str))[0]
    r, k = xc.max() + 1, yc.max() + 1
    if r < 2 or k < 2:
        return 0.0
    n = len(xc)
    table = np.bincount(xc * k + yc, minlength=r * k).reshape(r, k).astype(float)
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / n
    chi2 = float(((table - expected) ** 2 / np.where(expected > 0, expected, 1)).sum())
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / max(n - 1, 1))
    rc = r - (r - 1) ** 2 / max(n - 1, 1)
    kc = k - (k - 1) ** 2 / max(n - 1, 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else 0.0


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Correlation ratio eta in [0, 1]: how much of ``values``' variance the categories explain."""
    mask = categories.notna() & values.notna()
    if mask.sum() < 2:
        return float("nan")
    codes = pd.factorize(categories[mask].astype(str))[0]
    v = values[mask].to_numpy(dtype=float)
    total = float(((v - v.mean()) ** 2).sum())
    if total == 0:
        return 0.0
    cnt = np.bincount(codes)
    means = np.bincount(codes, weights=v) / np.maximum(cnt, 1)
    between = float((cnt * (means - v.mean()) ** 2).sum())
    return float(np.sqrt(between / total))


def spearman_abs(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if mask.sum() < 3:
        return float("nan")
    rho = a[mask].rank().corr(b[mask].rank())
    return float(abs(rho)) if pd.notna(rho) else 0.0


def association(a: pd.Series, a_kind: str, b: pd.Series, b_kind: str) -> tuple[float, str]:
    """Association of two series given their kinds ('num' or 'cat'), with the method name."""
    if a_kind == "num" and b_kind == "num":
        return spearman_abs(a, b), "spearman"
    if a_kind == "cat" and b_kind == "cat":
        return cramers_v(a, b), "cramers_v"
    if a_kind == "cat":
        return correlation_ratio(a, b), "eta"
    return correlation_ratio(b, a), "eta"


# ---------------------------------------------------------------------- EDA container
@dataclass
class EDAReport:
    profile: DatasetProfile
    target: str | None
    target_kind: str | None
    target_association: pd.DataFrame
    associations: pd.DataFrame
    target_by_feature: dict[str, pd.DataFrame] = field(default_factory=dict)
    missingness: pd.DataFrame = field(default_factory=pd.DataFrame)
    qc: Any = None
    analysed: pd.DataFrame | None = field(default=None, repr=False)  # typed sample used for charts
    skipped: dict[str, str] = field(default_factory=dict)  # columns that could not be analysed

    def summary(self) -> str:
        p = self.profile
        types = pd.Series({c: q.semantic_type for c, q in p.columns.items()}).value_counts()
        lines = [
            f"Rows: {p.n_rows:,}  Columns: {p.n_cols}  Duplicate rows: {p.n_duplicate_rows}",
            "Column types: " + ", ".join(f"{k}={v}" for k, v in types.items()),
        ]
        if self.target is not None:
            lines.append(f"Target '{self.target}' ({self.target_kind})")
            top = self.target_association.head(5)
            if not top.empty:
                lines.append(
                    "Top associations: "
                    + ", ".join(
                        f"{r.feature} ({r.strength:.2f}, {r.method})" for r in top.itertuples()
                    )
                )
        if self.qc is not None:
            bad = [i for i in self.qc.report_.items if i.status != "ok"]
            lines.append(
                f"QC: {len(bad)} warnings/errors, quality score {self.qc.data_quality_score_}/100"
            )
        return "\n".join(lines)

    def figures(
        self, max_distributions: int = 24, max_target_plots: int = 8
    ) -> list[tuple[str, Any]]:
        from mlkit.viz import eda_charts as ec

        figs: list[tuple[str, Any]] = [
            ("Column types and missing values", ec.plot_column_overview(self.profile))
        ]
        if self.analysed is not None:
            figs.append(
                (
                    "Distributions",
                    ec.plot_distributions(
                        self.analysed, self.profile, max_columns=max_distributions
                    ),
                )
            )
        if not self.missingness.empty:
            figs.append(("Missingness", ec.plot_missingness(self)))
        if not self.target_association.empty:
            figs.append(("Target association", ec.plot_target_association(self)))
            for col in self.target_association["feature"].head(max_target_plots):
                if col in self.target_by_feature:
                    figs.append((f"Target by {col}", ec.plot_target_by_feature(self, col)))
        if not self.associations.empty and len(self.associations) > 1:
            figs.append(("Feature associations", ec.plot_association_matrix(self)))
        return figs

    def to_html(
        self, output_path: str | Path | None = None, title: str = "Exploratory Data Analysis"
    ) -> str:
        from mlkit.viz.eda_report import render_eda_html

        return render_eda_html(self, output_path=output_path, title=title)


# ------------------------------------------------------------------------ analysis
def _typed_frame(
    df: pd.DataFrame, profile: DatasetProfile
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """Columns converted to analysable values, with kind 'num' or 'cat' (skips ids/constants).

    Conversions go through :func:`apply_coercion`, exactly as in preprocessing. A column that
    still cannot be converted is skipped and reported (third return value), never raised.
    """
    out: dict[str, pd.Series] = {}
    kinds: dict[str, str] = {}
    skipped: dict[str, str] = {}
    by_name = {str(c): c for c in df.columns}
    for name, p in profile.columns.items():
        if p.semantic_type in _SKIP or name not in by_name:
            continue
        try:
            s = df[by_name[name]]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            conv = apply_coercion(s, p.semantic_type, p.coercion)
            if p.semantic_type == "datetime":
                secs = pd.Series(np.nan, index=s.index, dtype=float)
                ok = conv.notna()
                secs[ok] = (
                    conv[ok].astype("datetime64[ns]").astype("int64") / 1e9
                )  # seconds since epoch
                out[name], kinds[name] = secs, "num"
            elif p.semantic_type == "text":
                out[name], kinds[name] = conv.astype("string").str.len().astype(float), "num"
            elif p.semantic_type == "numeric":
                out[name], kinds[name] = pd.to_numeric(conv, errors="coerce").astype(float), "num"
            elif pd.api.types.is_numeric_dtype(conv) and p.semantic_type in ("binary", "boolean"):
                out[name], kinds[name] = conv.astype(float), "cat"
            else:
                try:
                    obj = conv.astype(object)
                except (TypeError, ValueError):
                    obj = conv.map(str)
                out[name], kinds[name] = (
                    obj.where(conv.notna(), np.nan).map(
                        lambda v: (
                            v if isinstance(v, (str, float, int, bool)) or v is None else str(v)
                        )
                    ),
                    "cat",
                )
        except Exception as exc:  # noqa: BLE001 - robustness contract
            skipped[name] = f"{type(exc).__name__}: {str(exc)[:100]}"
    return pd.DataFrame(out, index=df.index), kinds, skipped


def _safe_association(a: pd.Series, a_kind: str, b: pd.Series, b_kind: str) -> tuple[float, str]:
    try:
        return association(a, a_kind, b, b_kind)
    except Exception:  # noqa: BLE001 - one odd column must not stop the EDA
        return float("nan"), "error"


def _target_kind(y: pd.Series, task_type: str | None) -> tuple[str, str]:
    """('binary'|'multiclass'|'regression', analysis kind 'cat'|'num')."""
    nn = y.dropna()
    if task_type == "regression" or (
        task_type is None and pd.api.types.is_numeric_dtype(nn) and nn.nunique() > 20
    ):
        return "regression", "num"
    return ("binary" if nn.nunique() == 2 else "multiclass"), "cat"


def _target_value(y: pd.Series, kind: str) -> tuple[pd.Series, str]:
    """Numeric series to average per bucket: positive-class indicator (binary) or the value."""
    if kind == "regression":
        return pd.to_numeric(y, errors="coerce"), "mean target"
    if kind == "binary":
        classes = sorted(y.dropna().astype(str).unique())
        pos = "1" if "1" in classes else classes[-1]
        return (y.astype(str) == pos).astype(float).where(y.notna()), f"rate of '{pos}'"
    top = y.astype(str).value_counts().index[0]
    return (y.astype(str) == top).astype(float).where(y.notna()), f"share of '{top}'"


def _buckets(x: pd.Series, kind: str, max_categories: int, n_bins: int) -> pd.Series:
    if kind == "num":
        valid = x.dropna()
        if valid.nunique() > n_bins:
            try:
                b = pd.qcut(x, q=n_bins, duplicates="drop")
                return b.astype(str).where(x.notna(), "(missing)")
            except ValueError:
                pass
        return x.round(4).astype(str).where(x.notna(), "(missing)")
    s = x.astype(str).where(x.notna(), "(missing)")
    top = s.value_counts().index[:max_categories]
    return s.where(s.isin(top) | (s == "(missing)"), "(other)")


def run_eda(
    df: pd.DataFrame,
    target: str | None = None,
    task_type: str | None = None,
    max_features: int = 30,
    max_categories: int = 20,
    n_bins: int = 10,
    sample_rows: int | None = 200_000,
    with_qc: bool = True,
    profile_options: ProfileOptions | None = None,
    random_state: int = 0,
    column_types: dict[str, str] | None = None,
) -> EDAReport:
    """Run the full EDA on ``df`` (optionally against ``target``, a column of ``df``)."""
    from mlkit.qc_eda.profile import profile_dataframe

    if target is not None and target not in df.columns:
        raise KeyError(f"Target column '{target}' not in data")
    data = df
    if sample_rows is not None and len(df) > sample_rows:
        data = df.sample(sample_rows, random_state=random_state)
    data, _dups = deduplicate_columns(data)
    X = data.drop(columns=[target]) if target is not None else data
    y = data[target] if target is not None else None
    if column_types:
        profile_options = ProfileOptions(
            **{**vars(profile_options or ProfileOptions()), "overrides": dict(column_types)}
        )

    profile = profile_dataframe(X, y=y, task_type=task_type, options=profile_options)
    qc = None
    if with_qc:
        from mlkit.core.config import TaskConfig
        from mlkit.qc_eda.qc import QualityCheckStep

        cfg = None
        if target is not None and task_type in ("classification", "regression"):
            cfg = TaskConfig(task_type=task_type, target=target, quality_gate=False)  # type: ignore[arg-type]
        qc = QualityCheckStep(enforce_gate=False, profile_options=profile_options).fit(
            X, y=y, cfg=cfg
        )

    typed, kinds, skipped = _typed_frame(X, profile)
    report = EDAReport(
        profile=profile,
        target=target,
        target_kind=None,
        target_association=pd.DataFrame(columns=["feature", "type", "method", "strength", "n"]),
        associations=pd.DataFrame(),
        qc=qc,
        analysed=typed,
        skipped=skipped,
    )

    ranked = list(typed.columns)
    if y is not None and y.notna().any() and y.dropna().nunique() > 1:
        t_kind, t_ana = _target_kind(y, task_type)
        report.target_kind = t_kind
        y_ana = (
            pd.to_numeric(y, errors="coerce")
            if t_ana == "num"
            else y.astype(object).where(y.notna(), np.nan)
        )
        rows = []
        for col in typed.columns:
            strength, method = _safe_association(typed[col], kinds[col], y_ana, t_ana)
            rows.append(
                {
                    "feature": col,
                    "type": profile.columns[col].semantic_type,
                    "method": method,
                    "strength": strength,
                    "n": int((typed[col].notna() & y.notna()).sum()),
                }
            )
        ta = (
            pd.DataFrame(rows, columns=["feature", "type", "method", "strength", "n"])
            .sort_values("strength", ascending=False, na_position="last")
            .reset_index(drop=True)
        )
        report.target_association = ta
        ranked = list(ta["feature"])

        y_val, label = _target_value(y, t_kind)
        for col in ranked[:max_features]:
            b = _buckets(typed[col], kinds[col], max_categories, n_bins)
            g = (
                pd.DataFrame({"bucket": b, "y": y_val})
                .groupby("bucket", sort=False)["y"]
                .agg(["size", "mean"])
            )
            g = g.rename(columns={"size": "n", "mean": label})
            if kinds[col] == "num":
                order = [k for k in b.drop_duplicates() if k != "(missing)"]
                try:
                    order = sorted(order, key=lambda s: float(str(s).strip("([").split(",")[0]))
                except ValueError:
                    order = sorted(order)
                g = g.reindex(order + (["(missing)"] if "(missing)" in g.index else []))
            else:
                g = g.sort_values("n", ascending=False)
                tail = [
                    k for k in ("(other)", "(missing)") if k in g.index
                ]  # pooled buckets go last
                g = g.reindex([k for k in g.index if k not in tail] + tail)
            report.target_by_feature[col] = g.reset_index()

        miss_rows = []
        for col in typed.columns:
            m = typed[col].isna()
            if 0 < m.mean() < 1:
                miss_rows.append(
                    {
                        "feature": col,
                        "missing_rate": float(m.mean()),
                        f"{label} | missing": float(y_val[m].mean()),
                        f"{label} | present": float(y_val[~m].mean()),
                    }
                )
        if miss_rows:
            mdf = pd.DataFrame(miss_rows)
            mdf["difference"] = mdf.iloc[:, 2] - mdf.iloc[:, 3]
            report.missingness = mdf.sort_values("missing_rate", ascending=False).reset_index(
                drop=True
            )
    else:
        miss = typed.isna().mean()
        miss = miss[miss > 0]
        if len(miss):
            report.missingness = (
                miss.rename("missing_rate")
                .rename_axis("feature")
                .reset_index()
                .sort_values("missing_rate", ascending=False)
            )

    top = ranked[:max_features]
    if len(top) > 1:
        mat = pd.DataFrame(np.eye(len(top)), index=top, columns=top)
        for i, col_a in enumerate(top):
            for col_b in top[i + 1 :]:
                v, _ = _safe_association(typed[col_a], kinds[col_a], typed[col_b], kinds[col_b])
                mat.loc[col_a, col_b] = mat.loc[col_b, col_a] = v
        report.associations = mat
    return report
