"""
mlkit.viz.html_report
~~~~~~~~~~~~~~~~~~~~~
Standalone, zero-dependency HTML report generator for MLKit pipelines, QC checks,
model arenas, and drift reports.
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


def generate_html_report(
    reports: Report | list[Report],
    title: str = "MLKit Pipeline Execution Report",
    summary_stats: dict[str, Any] | None = None,
    figures: list[tuple[str, plt.Figure]] | None = None,
    output_path: str | Path | None = None,
) -> str:
    """
    Generate a clean, standalone, responsive HTML report.

    Parameters
    ----------
    reports : Report or list of Report
        One or more MLKit Report objects.
    title : str
        Report header title.
    summary_stats : dict, optional
        Key-value pairs to display as KPI cards (e.g. Data Quality Score, Model Score).
    figures : list of (caption, Figure), optional
        Matplotlib figures to embed as base64 images.
    output_path : str or Path, optional
        If provided, writes HTML directly to this file.

    Returns
    -------
    str : Full HTML document string.
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
    status_color = {"ok": "#2ecc71", "warn": "#f39c12", "error": "#e74c3c"}[overall_status]
    status_label = {"ok": "PASSED", "warn": "WARNINGS DETECTED", "error": "FAILED / BLOCKED"}[overall_status]

    # Embedded Figures HTML
    fig_html = ""
    if figures:
        fig_cards = []
        for caption, fig in figures:
            b64_img = _fig_to_base64(fig)
            fig_cards.append(f"""
            <div class="card chart-card">
                <h3>{caption}</h3>
                <div class="img-container">
                    <img src="{b64_img}" alt="{caption}" />
                </div>
            </div>
            """)
        fig_html = f"""
        <section class="section">
            <h2>Visualizations & Diagnostic Plots</h2>
            <div class="grid-2">
                {''.join(fig_cards)}
            </div>
        </section>
        """

    # KPI Summary Cards HTML
    kpi_cards = [
        f"""
        <div class="kpi-card" style="border-top: 4px solid {status_color};">
            <span class="kpi-label">Gatekeeper Status</span>
            <span class="kpi-value" style="color: {status_color}; font-size: 1.3rem;">{status_label}</span>
        </div>
        """,
        f"""
        <div class="kpi-card" style="border-top: 4px solid #2ecc71;">
            <span class="kpi-label">Passed Checks</span>
            <span class="kpi-value" style="color: #2ecc71;">{total_ok}</span>
        </div>
        """,
        f"""
        <div class="kpi-card" style="border-top: 4px solid #f39c12;">
            <span class="kpi-label">Warnings</span>
            <span class="kpi-value" style="color: #f39c12;">{total_warn}</span>
        </div>
        """,
        f"""
        <div class="kpi-card" style="border-top: 4px solid #e74c3c;">
            <span class="kpi-label">Errors</span>
            <span class="kpi-value" style="color: #e74c3c;">{total_error}</span>
        </div>
        """,
    ]

    if summary_stats:
        for k, v in summary_stats.items():
            kpi_cards.append(f"""
            <div class="kpi-card" style="border-top: 4px solid #3498db;">
                <span class="kpi-label">{k}</span>
                <span class="kpi-value" style="color: #2c3e50;">{v}</span>
            </div>
            """)

    # Report Sections HTML
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
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """)

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
            --text-main: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --ok: #2ecc71;
            --warn: #f39c12;
            --error: #e74c3c;
            --primary: #3b82f6;
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
            padding: 2rem 1rem;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid var(--border);
        }}
        header h1 {{
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-main);
        }}
        header p {{
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-top: 0.25rem;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .kpi-card {{
            background: var(--surface);
            padding: 1.25rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}
        .kpi-label {{
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }}
        .kpi-value {{
            font-size: 1.8rem;
            font-weight: 700;
        }}
        .section {{
            margin-bottom: 2.5rem;
        }}
        .section h2 {{
            font-size: 1.3rem;
            margin-bottom: 1rem;
            color: var(--text-main);
        }}
        .card {{
            background: var(--surface);
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        .card h3 {{
            font-size: 1.1rem;
            margin-bottom: 1rem;
            color: var(--text-main);
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
            background: #d4edda;
            color: #155724;
        }}
        .badge-warn {{
            background: #fff3cd;
            color: #856404;
        }}
        .badge-error {{
            background: #f8d7da;
            color: #721c24;
        }}
        .details {{
            margin-top: 0.35rem;
            font-size: 0.8rem;
            color: var(--text-muted);
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 1.5rem;
        }}
        .chart-card img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
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
            <h1>{title}</h1>
            <p>Generated by MLKit &bull; Standardized Tabular Machine Learning Pipeline Toolkit</p>
        </header>

        <div class="kpi-grid">
            {''.join(kpi_cards)}
        </div>

        {fig_html}

        <section class="section">
            <h2>Pipeline Checks & Gatekeeper Audit</h2>
            {''.join(reports_html)}
        </section>

        <footer>
            <span>MLKit &copy; Production ML Engine &bull; All calculations verified</span>
        </footer>
    </div>
</body>
</html>
"""
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html_content, encoding="utf-8")

    return html_content
