"""Standardized Report system supporting ok, warn, and error severities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from mlkit.core.errors import QualityGateError

Severity = Literal["ok", "warn", "error"]


@dataclass
class ReportItem:
    """Individual check result within a report."""

    name: str
    status: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


class Report:
    """
    Structured report container tracking pipeline quality and gate status.
    """

    def __init__(self, title: str = "Report") -> None:
        self.title = title
        self.items: list[ReportItem] = []

    def add(
        self,
        name: str,
        status: Severity,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> Report:
        """Add a check result to the report."""
        self.items.append(
            ReportItem(
                name=name,
                status=status,
                message=message,
                details=details or {},
            )
        )
        return self

    @property
    def has_errors(self) -> bool:
        """Check if any items reported an error."""
        return any(item.status == "error" for item in self.items)

    @property
    def has_warnings(self) -> bool:
        """Check if any items reported a warning."""
        return any(item.status == "warn" for item in self.items)

    @property
    def is_passed(self) -> bool:
        """True if there are zero errors."""
        return not self.has_errors

    def raise_if_failed(self, gate_name: str = "Gatekeeper") -> None:
        """Raise QualityGateError if any item in the report failed with 'error'."""
        errors = [item for item in self.items if item.status == "error"]
        if errors:
            err_msgs = [f"[{e.name}] {e.message}" for e in errors]
            raise QualityGateError(
                f"{gate_name} check failed with {len(errors)} error(s):\n  - "
                + "\n  - ".join(err_msgs)
            )

    def merge(self, other: Report) -> Report:
        """Merge items from another report into this one."""
        self.items.extend(other.items)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "is_passed": self.is_passed,
            "error_count": sum(1 for i in self.items if i.status == "error"),
            "warning_count": sum(1 for i in self.items if i.status == "warn"),
            "items": [item.to_dict() for item in self.items],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        lines = [f"=== {self.title} (Passed: {self.is_passed}) ==="]
        for item in self.items:
            icon = "✅" if item.status == "ok" else ("⚠️" if item.status == "warn" else "❌")
            lines.append(f"  {icon} [{item.name}]: {item.message}")
        return "\n".join(lines)
