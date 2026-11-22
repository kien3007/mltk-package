"""Standalone HTML rendering of an :class:`mlkit.qc_eda.eda.EDAReport`.

Every chart is paired with the table it was drawn from, so the numbers are always readable
(charts are static PNGs on a light surface; the page is explicitly light-themed to match).
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from mlkit.viz.html_report import _fig_to_base64

if TYPE_CHECKING:
    from mlkit.qc_eda.eda import EDAReport

_CSS = """
:root { color-scheme: light; --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
        --muted:#898781; --grid:#e1e0d9; --good:#0ca30c; --warn:#fab219; --crit:#d03b3b; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink);
       font: 14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 32px 0 8px; }
.sub { color: var(--ink2); margin: 0 0 16px; }
nav { display:flex; flex-wrap:wrap; gap:8px; margin: 12px 0 8px; }
nav a { color: var(--ink2); text-decoration:none; border:1px solid var(--grid); border-radius:999px;
        padding: 2px 10px; font-size: 13px; background: var(--surface); }
.card { background: var(--surface); border:1px solid rgba(11,11,11,.10); border-radius: 10px;
        padding: 16px; margin: 12px 0; overflow-x: auto; }
.kpis { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.kpi .v { font-size: 22px; font-weight: 650; font-variant-numeric: tabular-nums; }
.kpi .l { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
img { max-width: 100%; height: auto; display:block; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; font-variant-numeric: tabular-nums; }
th, td { text-align:left; padding: 5px 8px; border-bottom: 1px solid var(--grid); vertical-align: top; }
th { color: var(--ink2); font-weight: 600; position: sticky; top: 0; background: var(--surface); }
.badge { display:inline-block; padding: 0 7px; border-radius: 999px; font-size: 12px; font-weight:600; }
.b-ok { color:#006300; } .b-warn { color:#7a5200; } .b-error { color: var(--crit); }
details summary { cursor: pointer; color: var(--ink2); margin-top: 8px; }
"""

_ICON = {"ok": "✓ ok", "warn": "▲ warning", "error": "✕ error"}


def _table(df: pd.DataFrame, float_fmt: str = "{:.4g}", max_rows: int = 200) -> str:
    if df is None or df.empty:
        return "<p class='sub'>Nothing to show.</p>"
    shown = df.head(max_rows).copy()
    for c in shown.columns:
        if pd.api.types.is_float_dtype(shown[c]):
            shown[c] = shown[c].map(lambda v: "" if pd.isna(v) else float_fmt.format(v))
    note = f"<p class='sub'>Showing {max_rows} of {len(df)} rows.</p>" if len(df) > max_rows else ""
    return shown.to_html(index=False, escape=True, border=0) + note


def _img(fig: plt.Figure, alt: str) -> str:
    src = _fig_to_base64(fig)
    plt.close(fig)
    return f"<img src='{src}' alt='{html.escape(alt)}'>"


def render_eda_html(
    report: EDAReport, output_path: str | Path | None = None, title: str = "EDA"
) -> str:
    from mlkit.viz import eda_charts as ec

    p = report.profile
    types = pd.Series({c: q.semantic_type for c, q in p.columns.items()}).value_counts()
    kpis = [
        ("rows", f"{p.n_rows:,}"),
        ("columns", f"{p.n_cols}"),
        ("duplicate rows", f"{p.n_duplicate_rows:,}"),
        ("memory", f"{p.memory_mb:.1f} MB"),
    ]
    if report.qc is not None:
        kpis.append(("quality score", f"{report.qc.data_quality_score_:.0f} / 100"))
    if report.target is not None:
        kpis.append(("target", f"{html.escape(str(report.target))} · {report.target_kind}"))

    sections: list[tuple[str, str, str]] = []
    overview = (
        "<div class='card kpis'>"
        + "".join(
            f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{html.escape(k)}</div></div>"
            for k, v in kpis
        )
        + "</div>"
    )
    overview += f"<div class='card'>{_img(ec.plot_column_overview(p), 'Column types and missing rates')}</div>"
    overview += "<div class='card'><h3>Columns</h3>" + _table(p.to_frame()) + "</div>"
    overview += (
        "<p class='sub'>Types: "
        + ", ".join(f"{html.escape(k)}={v}" for k, v in types.items())
        + "</p>"
    )
    sections.append(("overview", "Overview", overview))

    if report.qc is not None:
        rows = "".join(
            f"<tr><td><span class='badge b-{i.status}'>{_ICON[i.status]}</span></td>"
            f"<td>{html.escape(i.name)}</td><td>{html.escape(i.message)}</td></tr>"
            for i in sorted(
                report.qc.report_.items, key=lambda i: {"error": 0, "warn": 1, "ok": 2}[i.status]
            )
        )
        qc_html = f"<div class='card'><table><tr><th>status</th><th>check</th><th>finding</th></tr>{rows}</table></div>"
        issues = report.qc.issues_frame_
        if not issues.empty:
            qc_html += "<div class='card'><h3>Column issues</h3>" + _table(issues) + "</div>"
        sections.append(("quality", "Data quality", qc_html))

    if report.analysed is not None and len(report.analysed.columns):
        dist = f"<div class='card'>{_img(ec.plot_distributions(report.analysed, p), 'Distributions')}</div>"
        sections.append(("distributions", "Distributions", dist))

    if not report.target_association.empty:
        tgt = f"<div class='card'>{_img(ec.plot_target_association(report), 'Target association')}"
        tgt += (
            "<details><summary>Table</summary>"
            + _table(report.target_association)
            + "</details></div>"
        )
        for col in report.target_association["feature"].head(8):
            if col in report.target_by_feature:
                tgt += f"<div class='card'>{_img(ec.plot_target_by_feature(report, col), f'Target by {col}')}"
                tgt += (
                    "<details><summary>Table</summary>"
                    + _table(report.target_by_feature[col])
                    + "</details></div>"
                )
        sections.append(("target", "Target analysis", tgt))

    if not report.missingness.empty:
        miss = f"<div class='card'>{_img(ec.plot_missingness(report), 'Missingness')}"
        miss += (
            "<details><summary>Table</summary>" + _table(report.missingness) + "</details></div>"
        )
        sections.append(("missing", "Missing values", miss))

    if not report.associations.empty and len(report.associations) > 1:
        assoc = (
            f"<div class='card'>{_img(ec.plot_association_matrix(report), 'Feature associations')}"
        )
        assoc += (
            "<details><summary>Table</summary>"
            + _table(report.associations.reset_index(names="feature"))
            + "</details></div>"
        )
        sections.append(("associations", "Associations", assoc))

    nav = (
        "<nav>"
        + "".join(f"<a href='#{sid}'>{html.escape(t)}</a>" for sid, t, _ in sections)
        + "</nav>"
    )
    body = "".join(
        f"<section id='{sid}'><h2>{html.escape(t)}</h2>{content}</section>"
        for sid, t, content in sections
    )
    doc = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body><main>"
        f"<h1>{html.escape(title)}</h1><p class='sub'>Generated by mlkit · {p.n_rows:,} rows × {p.n_cols} columns"
        "</p>" + nav + body + "</main></body></html>"
    )
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(doc, encoding="utf-8")
    return doc
