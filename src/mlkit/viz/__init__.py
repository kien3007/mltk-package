"""MLKit Visualization module — charts and standalone HTML reports for every pipeline stage."""

from mlkit.viz.charts import (
    plot_arena_comparison,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_drift_psi,
    plot_feature_importance,
    plot_missing_rates,
    plot_roc_curve,
    plot_target_distribution,
)
from mlkit.viz.html_report import generate_html_report

__all__ = [
    "generate_html_report",
    "plot_arena_comparison",
    "plot_calibration_curve",
    "plot_confusion_matrix",
    "plot_correlation_heatmap",
    "plot_drift_psi",
    "plot_feature_importance",
    "plot_missing_rates",
    "plot_roc_curve",
    "plot_target_distribution",
]
