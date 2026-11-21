"""
mlkit.viz.charts
~~~~~~~~~~~~~~~~
Production-ready visualization helpers for every stage of the mlkit pipeline.

All functions:
- Accept native mlkit objects (Report, ModelArena, DriftMonitor) OR raw data.
- Return a ``matplotlib.figure.Figure`` so callers can ``fig.savefig(...)`` or display inline.
- Never call ``plt.show()`` internally (caller decides when / how to render).
- Backend is set to ``Agg`` by default so the library is safe in headless environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from mlkit.core.report import Report
    from mlkit.modelling.arena import ModelArena

matplotlib.use("Agg")  # headless-safe; call before any other plt usage

# ---------------------------------------------------------------------------
# Shared style helper
# ---------------------------------------------------------------------------
_PALETTE = {"ok": "#2ecc71", "warn": "#f39c12", "error": "#e74c3c", "neutral": "#3498db"}


def _apply_style(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


# ---------------------------------------------------------------------------
# 1. QC / EDA ─ Missing Rate Bar Chart
# ---------------------------------------------------------------------------
def plot_missing_rates(
    df: pd.DataFrame,
    threshold: float = 0.5,
    top_n: int = 30,
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Horizontal bar chart of per-column missing rates.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data.
    threshold : float
        Red dashed line marking the acceptable missing threshold.
    top_n : int
        Show at most this many columns (sorted by missing rate descending).
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    missing = df.isna().mean().sort_values(ascending=False).head(top_n)
    colors = [_PALETTE["error"] if v >= threshold else _PALETTE["ok"] for v in missing.values]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(missing.index[::-1], missing.values[::-1], color=colors[::-1])
    ax.axvline(x=threshold, color="gray", linestyle="--", linewidth=1.2, label=f"Threshold ({threshold:.0%})")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _apply_style(ax, "Missing Rate per Column", xlabel="Missing Rate")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. QC / EDA ─ Target Distribution
# ---------------------------------------------------------------------------
def plot_target_distribution(
    y: pd.Series | np.ndarray,
    task_type: str = "classification",
    figsize: tuple[int, int] = (8, 4),
) -> plt.Figure:
    """
    Bar chart (classification) or histogram (regression) of the target variable.

    Parameters
    ----------
    y : array-like
        Target series.
    task_type : str
        ``'classification'`` or ``'regression'``.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_s = pd.Series(y)
    fig, ax = plt.subplots(figsize=figsize)

    if task_type == "classification":
        counts = y_s.value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color=_PALETTE["neutral"])
        _apply_style(ax, "Target Class Distribution", xlabel="Class", ylabel="Count")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 0.5, str(v), ha="center", fontsize=9)
    else:
        ax.hist(y_s.dropna(), bins=40, color=_PALETTE["neutral"], edgecolor="white")
        _apply_style(ax, "Target Distribution (Regression)", xlabel="Value", ylabel="Frequency")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. QC / EDA ─ Correlation Heatmap
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(
    df: pd.DataFrame,
    method: str = "pearson",
    top_n: int = 25,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Heatmap of pairwise feature correlations (numeric columns only).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset (numeric columns only are used).
    method : str
        Correlation method: ``'pearson'``, ``'spearman'``, or ``'kendall'``.
    top_n : int
        Maximum number of features to include.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    num_df = df.select_dtypes(include=[np.number]).iloc[:, :top_n]
    corr = num_df.corr(method=method)

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=len(corr) <= 15,
        fmt=".2f",
        linewidths=0.3,
        ax=ax,
        square=True,
    )
    _apply_style(ax, f"Feature Correlation Heatmap ({method.capitalize()})")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Modelling ─ Arena Comparison Bar Chart
# ---------------------------------------------------------------------------
def plot_arena_comparison(
    arena: ModelArena | pd.DataFrame,
    metric: str | None = None,
    figsize: tuple[int, int] = (9, 4),
) -> plt.Figure:
    """
    Horizontal bar chart comparing model scores from a fitted ``ModelArena``.

    Parameters
    ----------
    arena : ModelArena or pd.DataFrame
        A fitted ``mlkit.modelling.ModelArena`` instance, or its ``comparison_table_``
        DataFrame directly.
    metric : str, optional
        Column name to plot. Auto-detected from ``arena.comparison_table_`` if not given.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if isinstance(arena, pd.DataFrame):
        df = arena.copy()
    else:
        df = arena.comparison_table_.copy()

    if metric is None:
        metric = "Score"

    if metric not in df.columns:
        raise ValueError(f"Column '{metric}' not found in comparison table. Available: {list(df.columns)}")

    df = df.sort_values(metric, ascending=True)
    colors = [_PALETTE["error"] if "Baseline" in str(m) else _PALETTE["neutral"] for m in df["Model"]]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(df["Model"], df[metric], color=colors)

    for bar, val in zip(bars, df[metric], strict=False):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=9)

    _apply_style(ax, f"Model Arena — {metric}", xlabel=metric)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Modelling ─ Feature Importance
# ---------------------------------------------------------------------------
def plot_feature_importance(
    arena: ModelArena | Any,
    feature_names: list[str] | None = None,
    top_n: int = 25,
    figsize: tuple[int, int] = (9, 6),
) -> plt.Figure:
    """
    Horizontal bar chart of feature importances from the best model in ``ModelArena``.

    Parameters
    ----------
    arena : ModelArena or sklearn-compatible estimator
        A fitted ``ModelArena`` or any estimator with ``feature_importances_``.
    feature_names : list[str], optional
        Column names. Required if ``arena`` is a plain estimator.
    top_n : int
        Show at most this many features.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Extract importances
    importances = None
    names: list[str] = []

    if hasattr(arena, "feature_importances_"):
        importances = np.asarray(arena.feature_importances_)
    elif hasattr(arena, "best_estimator_") and hasattr(arena.best_estimator_, "feature_importances_"):
        importances = np.asarray(arena.best_estimator_.feature_importances_)
    else:
        raise ValueError("No 'feature_importances_' found on the provided object.")

    # Feature names
    if feature_names is not None:
        names = list(feature_names)
    elif hasattr(arena, "best_estimator_") and hasattr(arena.best_estimator_, "feature_names_in_"):
        names = list(arena.best_estimator_.feature_names_in_)
    elif hasattr(arena, "feature_names_in_"):
        names = list(arena.feature_names_in_)
    else:
        names = [f"feature_{i}" for i in range(len(importances))]

    # Align lengths
    min_len = min(len(importances), len(names))
    importances = importances[:min_len]
    names = names[:min_len]

    # Sort and limit
    idx = np.argsort(importances)[::-1][:top_n]
    top_imp = importances[idx][::-1]
    top_names = [names[i] for i in idx][::-1]

    # Normalize to 0-1
    top_imp_norm = top_imp / (top_imp.sum() + 1e-10)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top_names, top_imp_norm, color=_PALETTE["neutral"])
    _apply_style(ax, f"Feature Importance (Top {top_n})", xlabel="Relative Importance")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Monitoring ─ PSI Drift Bar Chart
# ---------------------------------------------------------------------------
def plot_drift_psi(
    drift_report: Report,
    warn_threshold: float = 0.1,
    error_threshold: float = 0.2,
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Bar chart of PSI scores from a drift monitoring report.

    Parameters
    ----------
    drift_report : mlkit.core.report.Report
        The ``Report`` returned by ``DriftMonitor.monitor()``.
    warn_threshold : float
        PSI threshold for warn level (orange).
    error_threshold : float
        PSI threshold for error level (red).
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    rows = []
    for item in drift_report.items:
        psi = item.details.get("psi")
        if psi is not None:
            col_name = item.name.replace("drift_", "")
            rows.append({"Feature": col_name, "PSI": float(psi), "Status": item.status})

    if not rows:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No numeric drift data found in report.", ha="center", va="center", transform=ax.transAxes)
        return fig

    psi_df = pd.DataFrame(rows).sort_values("PSI", ascending=True)
    colors = []
    for _, r in psi_df.iterrows():
        if r["PSI"] >= error_threshold:
            colors.append(_PALETTE["error"])
        elif r["PSI"] >= warn_threshold:
            colors.append(_PALETTE["warn"])
        else:
            colors.append(_PALETTE["ok"])

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(psi_df["Feature"], psi_df["PSI"], color=colors)
    ax.axvline(x=warn_threshold, color=_PALETTE["warn"], linestyle="--", linewidth=1.2, label=f"Warn ({warn_threshold})")
    ax.axvline(x=error_threshold, color=_PALETTE["error"], linestyle="--", linewidth=1.2, label=f"Error ({error_threshold})")
    _apply_style(ax, "Feature Drift — PSI Score", xlabel="PSI")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Modelling ─ Confusion Matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    labels: list | None = None,
    normalize: bool = True,
    figsize: tuple[int, int] = (6, 5),
) -> plt.Figure:
    """
    Heatmap confusion matrix for classification tasks.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    labels : list, optional
        Display labels for each class.
    normalize : bool
        If True, show proportions (row-wise); otherwise raw counts.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    tick_labels = labels if labels is not None else sorted(set(y_true))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ax=ax,
        linewidths=0.5,
    )
    _apply_style(ax, "Confusion Matrix", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8. Modelling ─ ROC Curve
# ---------------------------------------------------------------------------
def plot_roc_curve(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    figsize: tuple[int, int] = (6, 5),
) -> plt.Figure:
    """
    ROC curve with AUC annotation for binary classification.

    Parameters
    ----------
    y_true : array-like
        Binary ground-truth labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class (1-D or shape ``(n, 2)``).
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color=_PALETTE["neutral"], lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.fill_between(fpr, tpr, alpha=0.1, color=_PALETTE["neutral"])
    _apply_style(ax, "ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 9. Modelling ─ Calibration Curve (Reliability Diagram)
# ---------------------------------------------------------------------------
def plot_calibration_curve(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    n_bins: int = 10,
    figsize: tuple[int, int] = (6, 5),
) -> plt.Figure:
    """
    Plot calibration curve (reliability diagram) for predicted probabilities.

    Parameters
    ----------
    y_true : array-like
        Binary target labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    n_bins : int
        Number of probability bins.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    brier = brier_score_loss(y_true, y_prob)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.6)
    ax.plot(prob_pred, prob_true, "s-", color=_PALETTE["neutral"], label=f"Model (Brier={brier:.4f})")
    _apply_style(
        ax,
        "Probability Calibration Curve",
        xlabel="Mean Predicted Probability",
        ylabel="Fraction of Positives",
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig

