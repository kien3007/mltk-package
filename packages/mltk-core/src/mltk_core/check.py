"""Check, CheckResult and Report classes for data quality and validation."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import pandas as pd

from mltk_core.exceptions import SchemaError


@dataclass
class CheckResult:
    """The outcome of a single validation or quality check."""

    name: str
    level: Literal["INFO", "WARNING", "ERROR"]
    passed: bool
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
        }


class Check(ABC):
    """Abstract base check."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the check."""

    @abstractmethod
    def run(self, df: pd.DataFrame) -> CheckResult:
        """Execute check against DataFrame and return CheckResult."""


class Report:
    """Aggregated report of multiple checks."""

    def __init__(self, results: list[CheckResult] | None = None, title: str = "Quality & Validation Report") -> None:
        self.results: list[CheckResult] = results or []
        self.title: str = title
        self.created_at: str = datetime.now(timezone.utc).isoformat()

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    @property
    def passed(self) -> bool:
        """Returns True if no check with level 'ERROR' failed."""
        return not any(r.level == "ERROR" and not r.passed for r in self.results)

    @property
    def has_errors(self) -> bool:
        return any(r.level == "ERROR" and not r.passed for r in self.results)

    @property
    def has_warnings(self) -> bool:
        return any(r.level == "WARNING" and not r.passed for r in self.results)

    def to_frame(self) -> pd.DataFrame:
        """Convert check results into a pandas DataFrame."""
        rows = [
            {
                "name": r.name,
                "level": r.level,
                "passed": r.passed,
                "message": r.message,
                "details": json.dumps(r.details, default=str),
            }
            for r in self.results
        ]
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "created_at": self.created_at,
            "passed": self.passed,
            "total_checks": len(self.results),
            "errors": sum(1 for r in self.results if r.level == "ERROR" and not r.passed),
            "warnings": sum(1 for r in self.results if r.level == "WARNING" and not r.passed),
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_html(self, title: str | None = None) -> str:
        """Generate a clean, standalone modern HTML report."""
        report_title = title or self.title
        summary = self.to_dict()

        status_badge = (
            '<span style="background-color:#10b981; color:white; padding:4px 10px; border-radius:4px; font-weight:600;">PASSED</span>'
            if self.passed
            else '<span style="background-color:#ef4444; color:white; padding:4px 10px; border-radius:4px; font-weight:600;">FAILED</span>'
        )

        rows_html = []
        for r in self.results:
            badge_color = "#10b981" if r.passed else ("#ef4444" if r.level == "ERROR" else "#f59e0b")
            status_text = "PASS" if r.passed else "FAIL"
            details_str = json.dumps(r.details, indent=2, default=str) if r.details else "-"
            rows_html.append(
                f"""<tr>
                    <td style="padding:10px; border-bottom:1px solid #e5e7eb; font-weight:500;">{r.name}</td>
                    <td style="padding:10px; border-bottom:1px solid #e5e7eb;"><span style="color:{badge_color}; font-weight:600;">{status_text}</span></td>
                    <td style="padding:10px; border-bottom:1px solid #e5e7eb; color:#6b7280;">{r.level}</td>
                    <td style="padding:10px; border-bottom:1px solid #e5e7eb;">{r.message}</td>
                    <td style="padding:10px; border-bottom:1px solid #e5e7eb;"><pre style="margin:0; font-size:12px; background:#f9fafb; padding:4px 8px; border-radius:4px;">{details_str}</pre></td>
                </tr>"""
            )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{report_title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f3f4f6; color: #1f2937; margin:0; padding:24px; }}
        .card {{ background: white; border-radius: 8px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); max-width: 1000px; margin: 0 auto; }}
        .stats {{ display: flex; gap: 16px; margin: 20px 0; }}
        .stat-box {{ flex: 1; padding: 16px; border-radius: 6px; background: #f9fafb; border: 1px solid #e5e7eb; text-align: center; }}
        .stat-val {{ font-size: 24px; font-weight: 700; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 14px; }}
        th {{ text-align: left; padding: 12px 10px; background: #f9fafb; border-bottom: 2px solid #e5e7eb; }}
    </style>
</head>
<body>
    <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <h1 style="margin:0; font-size:24px;">{report_title}</h1>
            <div>{status_badge}</div>
        </div>
        <p style="color:#6b7280; margin: 6px 0 0 0; font-size:13px;">Generated at: {self.created_at}</p>
        
        <div class="stats">
            <div class="stat-box"><div class="stat-val">{summary["total_checks"]}</div><div style="color:#6b7280; font-size:13px;">Total Checks</div></div>
            <div class="stat-box"><div class="stat-val" style="color:#ef4444;">{summary["errors"]}</div><div style="color:#6b7280; font-size:13px;">Errors</div></div>
            <div class="stat-box"><div class="stat-val" style="color:#f59e0b;">{summary["warnings"]}</div><div style="color:#6b7280; font-size:13px;">Warnings</div></div>
        </div>

        <table>
            <thead>
                <tr><th>Check Name</th><th>Status</th><th>Level</th><th>Message</th><th>Details</th></tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
</body>
</html>"""
        return html

    def raise_if_failed(self) -> None:
        """Raise SchemaError if any error-level check failed."""
        if self.has_errors:
            failed_msgs = [f"[{r.name}] {r.message}" for r in self.results if r.level == "ERROR" and not r.passed]
            raise SchemaError(f"Validation failed with {len(failed_msgs)} errors:\n" + "\n".join(failed_msgs))
