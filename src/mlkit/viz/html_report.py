"""
mlkit.viz.html_report
~~~~~~~~~~~~~~~~~~~~~
Standalone, interactive multi-tab HTML report dashboard for MLKit pipelines,
QC checks, model arenas, business curves, and drift reports.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

    from mlkit.core.report import Report


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib Figure to a base64 encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _classify_fig_tab(caption: str) -> str:
    """Classify figure into a dashboard category tab based on caption keywords."""
    c = caption.lower()
    if any(k in c for k in ["missing", "distribution", "correlation", "target", "eda", "quality"]):
        return "tab-eda"
    if any(k in c for k in ["arena", "benchmark", "learning curve"]):
        return "tab-arena"
    if any(k in c for k in ["roc", "precision", "recall", "confusion", "gain", "lift"]):
        return "tab-business"
    if any(k in c for k in ["importance", "shap", "calibration", "explain"]):
        return "tab-xai"
    if any(k in c for k in ["drift", "psi", "monitor"]):
        return "tab-drift"
    return "tab-eda"


def generate_html_report(
    reports: Report | list[Report],
    title: str = "MLKit Pipeline Execution Report",
    summary_stats: dict[str, Any] | None = None,
    figures: list[tuple[str, plt.Figure]] | None = None,
    output_path: str | Path | None = None,
) -> str:
    """
    Generate an interactive, standalone multi-tab HTML dashboard report.
    """
    report_list = [reports] if not isinstance(reports, list) else reports

    total_ok = 0
    total_warn = 0
    total_error = 0
    for r in report_list:
        for it in r.items:
            if it.status == "ok":
                total_ok += 1
            elif it.status == "warn":
                total_warn += 1
            elif it.status == "error":
                total_error += 1

    overall_status = "error" if total_error > 0 else ("warn" if total_warn > 0 else "ok")
    status_color = {"ok": "#10b981", "warn": "#f59e0b", "error": "#ef4444"}[overall_status]
    status_label = {"ok": "PASSED", "warn": "WARNINGS DETECTED", "error": "FAILED / BLOCKED"}[
        overall_status
    ]

    # Categorize embedded figures into tabs
    tab_figures: dict[str, list[str]] = {
        "tab-all": [],
        "tab-eda": [],
        "tab-arena": [],
        "tab-business": [],
        "tab-xai": [],
        "tab-drift": [],
    }

    if figures:
        for caption, fig in figures:
            b64_img = _fig_to_base64(fig)
            card_html = f"""
            <div class="card chart-card">
                <h3>{caption}</h3>
                <div class="img-container">
                    <img src="{b64_img}" alt="{caption}" loading="lazy" />
                </div>
            </div>
            """
            tab_figures["tab-all"].append(card_html)
            tab_category = _classify_fig_tab(caption)
            tab_figures[tab_category].append(card_html)

    # KPI Summary Cards HTML
    kpi_cards = [
        f"""
        <div class="kpi-card" style="border-top: 4px solid {status_color};">
            <span class="kpi-label">Gatekeeper Status</span>
            <span class="kpi-value" style="color: {status_color}; font-size: 1.3rem;">{status_label}</span>
        </div>
        """,
        f"""
        <div class="kpi-card" style="border-top: 4px solid #10b981;">
            <span class="kpi-label">Passed Checks</span>
            <span class="kpi-value" style="color: #10b981;">{total_ok}</span>
        </div>
        """,
        f"""
        <div class="kpi-card" style="border-top: 4px solid #f59e0b;">
            <span class="kpi-label">Warnings</span>
            <span class="kpi-value" style="color: #f59e0b;">{total_warn}</span>
        </div>
        """,
        f"""
        <div class="kpi-card" style="border-top: 4px solid #ef4444;">
            <span class="kpi-label">Errors</span>
            <span class="kpi-value" style="color: #ef4444;">{total_error}</span>
        </div>
        """,
    ]

    if summary_stats:
        for k, v in summary_stats.items():
            kpi_cards.append(f"""
            <div class="kpi-card" style="border-top: 4px solid #3b82f6;">
                <span class="kpi-label">{k}</span>
                <span class="kpi-value" style="color: #1e293b;">{v}</span>
            </div>
            """)

    # Report Audit Tables HTML
    reports_html = []
    for r in report_list:
        rows = []
        for it in r.items:
            icon = "✅" if it.status == "ok" else ("⚠️" if it.status == "warn" else "❌")
            badge_class = f"badge badge-{it.status}"
            details_str = ""
            if it.details:
                details_items = [f"<strong>{dk}:</strong> {dv}" for dk, dv in it.details.items()]
                details_str = f"<div class='details'>{' &bull; '.join(details_items)}</div>"

            rows.append(f"""
            <tr>
                <td><span class="{badge_class}">{icon} {it.status.upper()}</span></td>
                <td><code>{it.name}</code></td>
                <td>
                    <div class="message">{it.message}</div>
                    {details_str}
                </td>
            </tr>
            """)

        reports_html.append(f"""
        <div class="card table-card">
            <h3>{r.title}</h3>
            <table class="report-table">
                <thead>
                    <tr>
                        <th style="width: 120px;">Status</th>
                        <th style="width: 220px;">Check Name</th>
                        <th>Message & Details</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </div>
        """)

    def _render_tab_grid(cards: list[str], empty_msg: str) -> str:
        if not cards:
            return f"<p class='empty-text'>{empty_msg}</p>"
        return f"<div class='grid-2'>{''.join(cards)}</div>"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg: #f8fafc;
            --surface: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --ok: #10b981;
            --warn: #f59e0b;
            --error: #ef4444;
            --primary: #2563eb;
            --primary-bg: #eff6ff;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        body {{
            background-color: var(--bg);
            color: var(--text-main);
            padding: 2rem 1.5rem;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1300px;
            margin: 0 auto;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1.25rem;
            border-bottom: 2px solid var(--border);
        }}
        header h1 {{
            font-size: 1.85rem;
            font-weight: 800;
            color: var(--text-main);
            letter-spacing: -0.02em;
        }}
        header p {{
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-top: 0.25rem;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .kpi-card {{
            background: var(--surface);
            padding: 1.25rem;
            border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }}
        .kpi-label {{
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }}
        .kpi-value {{
            font-size: 1.85rem;
            font-weight: 700;
        }}
        /* Tab Navigation */
        .tab-nav {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
            background: #e2e8f0;
            padding: 0.4rem;
            border-radius: 10px;
        }}
        .tab-btn {{
            background: transparent;
            border: none;
            outline: none;
            padding: 0.65rem 1.1rem;
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-muted);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .tab-btn:hover {{
            color: var(--text-main);
            background: rgba(255, 255, 255, 0.6);
        }}
        .tab-btn.active {{
            background: var(--surface);
            color: var(--primary);
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        .card {{
            background: var(--surface);
            border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        .card h3 {{
            font-size: 1.15rem;
            margin-bottom: 1rem;
            color: var(--text-main);
            font-weight: 700;
        }}
        .report-table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.9rem;
        }}
        .report-table th {{
            background: #f1f5f9;
            color: var(--text-muted);
            font-weight: 600;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
        }}
        .report-table td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
            vertical-align: top;
        }}
        .report-table tr:hover {{
            background-color: #f8fafc;
        }}
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.05em;
        }}
        .badge-ok {{
            background: #d1fae5;
            color: #065f46;
        }}
        .badge-warn {{
            background: #fef3c7;
            color: #92400e;
        }}
        .badge-error {{
            background: #fee2e2;
            color: #991b1b;
        }}
        .details {{
            margin-top: 0.35rem;
            font-size: 0.8rem;
            color: var(--text-muted);
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(540px, 1fr));
            gap: 1.5rem;
        }}
        .chart-card img {{
            width: 100%;
            height: auto;
            border-radius: 6px;
        }}
        .empty-text {{
            color: var(--text-muted);
            padding: 2rem;
            text-align: center;
            font-style: italic;
        }}
        footer {{
            margin-top: 3rem;
            text-align: center;
            font-size: 0.85rem;
            color: var(--text-muted);
            border-top: 1px solid var(--border);
            padding-top: 1.5rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>{title}</h1>
                <p>Standardized Tabular Machine Learning Pipeline Toolkit &bull; Verification &amp; Diagnostics</p>
            </div>
        </header>

        <div class="kpi-grid">
            {"".join(kpi_cards)}
        </div>

        <!-- Interactive Tab Navigation -->
        <nav class="tab-nav">
            <button class="tab-btn active" onclick="openTab('tab-all')">🖼️ All Visualizations ({len(tab_figures["tab-all"])})</button>
            <button class="tab-btn" onclick="openTab('tab-eda')">🔍 Data Quality &amp; EDA ({len(tab_figures["tab-eda"])})</button>
            <button class="tab-btn" onclick="openTab('tab-arena')">⚔️ Model Arena ({len(tab_figures["tab-arena"])})</button>
            <button class="tab-btn" onclick="openTab('tab-business')">🎯 Business &amp; Decision Curves ({len(tab_figures["tab-business"])})</button>
            <button class="tab-btn" onclick="openTab('tab-xai')">🧠 Explainability &amp; SHAP ({len(tab_figures["tab-xai"])})</button>
            <button class="tab-btn" onclick="openTab('tab-drift')">📡 Drift &amp; Stability ({len(tab_figures["tab-drift"])})</button>
            <button class="tab-btn" onclick="openTab('tab-audit')">📋 Gatekeeper Audit Log</button>
        </nav>

        <!-- Tab Contents -->
        <div id="tab-all" class="tab-content active">
            {_render_tab_grid(tab_figures["tab-all"], "No charts provided.")}
        </div>

        <div id="tab-eda" class="tab-content">
            {_render_tab_grid(tab_figures["tab-eda"], "No EDA charts in this run.")}
        </div>

        <div id="tab-arena" class="tab-content">
            {_render_tab_grid(tab_figures["tab-arena"], "No Model Arena charts in this run.")}
        </div>

        <div id="tab-business" class="tab-content">
            {_render_tab_grid(tab_figures["tab-business"], "No Business Decision curves generated.")}
        </div>

        <div id="tab-xai" class="tab-content">
            {_render_tab_grid(tab_figures["tab-xai"], "No Explainability / SHAP charts generated.")}
        </div>

        <div id="tab-drift" class="tab-content">
            {_render_tab_grid(tab_figures["tab-drift"], "No Drift Monitoring charts generated.")}
        </div>

        <div id="tab-audit" class="tab-content">
            <section class="section">
                {"".join(reports_html)}
            </section>
        </div>

        <footer>
            <span>MLKit Engine &copy; Production Ready &bull; Zero external frontend dependencies</span>
        </footer>
    </div>

    <script>
        function openTab(tabId) {{
            // Deactivate all tabs
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

            // Activate chosen tab
            document.getElementById(tabId).classList.add('active');
            event.currentTarget.classList.add('active');
        }}
    </script>
</body>
</html>
"""
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html_content, encoding="utf-8")

    return html_content
