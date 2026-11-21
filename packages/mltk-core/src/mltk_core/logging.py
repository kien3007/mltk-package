"""Structured JSON logging with contextvar run_id tracking."""

from __future__ import annotations

import contextvars
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

_CURRENT_RUN_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "mltk_current_run_id", default=""
)


def set_run_id(run_id: str | None = None) -> str:
    """Set or generate a new active run_id in the current execution context."""
    active_id = run_id or uuid.uuid4().hex[:12]
    _CURRENT_RUN_ID.set(active_id)
    return active_id


def get_run_id() -> str:
    """Retrieve the current active run_id, initializing one if absent."""
    active_id = _CURRENT_RUN_ID.get()
    if not active_id:
        active_id = set_run_id()
    return active_id


class JSONFormatter(logging.Formatter):
    """Formats log records into single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "run_id": get_run_id(),
            "msg": record.getMessage(),
        }

        # Add optional step and extra fields if attached to the record
        if hasattr(record, "step"):
            log_entry["step"] = getattr(record, "step")

        extra_data = getattr(record, "extra_data", None)
        if isinstance(extra_data, dict):
            log_entry["extra"] = extra_data

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def get_logger(name: str = "mltk", level: int = logging.INFO) -> logging.Logger:
    """Get or configure a structured JSON logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if already configured
    if not any(isinstance(h.formatter, JSONFormatter) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    return logger
