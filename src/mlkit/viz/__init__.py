"""MLKit Visualization module — rich diagnostic charts and standalone HTML dashboards."""

from mlkit.viz.charts import (
    plot_arena_comparison,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_cumulative_gains,
    plot_drift_psi,
    plot_feature_distribution_by_target,
    plot_feature_importance,
    plot_learning_curve,
    plot_lift_curve,
    plot_missing_rates,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_shap_summary,
    plot_target_distribution,
)
from mlkit.viz.eda_charts import (
    plot_association_matrix,
    plot_column_overview,
    plot_distributions,
    plot_missingness,
    plot_target_association,
    plot_target_by_feature,
)
from mlkit.viz.html_report import generate_html_report

__all__ = [
    "generate_html_report",
    "plot_association_matrix",
    "plot_column_overview",
    "plot_distributions",
    "plot_missingness",
    "plot_target_association",
    "plot_target_by_feature",
    "plot_arena_comparison",
    "plot_calibration_curve",
    "plot_confusion_matrix",
    "plot_correlation_heatmap",
    "plot_cumulative_gains",
    "plot_drift_psi",
    "plot_feature_distribution_by_target",
    "plot_feature_importance",
    "plot_learning_curve",
    "plot_lift_curve",
    "plot_missing_rates",
    "plot_precision_recall_curve",
    "plot_roc_curve",
    "plot_shap_summary",
    "plot_target_distribution",
]
