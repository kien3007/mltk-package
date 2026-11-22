"""Structured JSON logging with a ``run_id`` carried through a ContextVar (ADR-002)."""

from __future__ import annotations

import contextlib
import json
import logging
import sys
import uuid
from collections.abc import Iterator
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

_run_id: ContextVar[str | None] = ContextVar("mlkit_run_id", default=None)

_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()) | {"message"}


def get_run_id() -> str | None:
    """Return the run_id bound to the current context (or None)."""
    return _run_id.get()


@contextlib.contextmanager
def run_context(run_id: str | None = None) -> Iterator[str]:
    """Bind a ``run_id`` to every log record emitted inside this block."""
    rid = run_id or uuid.uuid4().hex[:12]
    token = _run_id.set(rid)
    try:
        yield rid
    finally:
        _run_id.reset(token)


class JsonFormatter(logging.Formatter):
    """Format each record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "run_id": get_run_id(),
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rid = get_run_id()
        prefix = f"[mlkit{':' + rid if rid else ''}]"
        return f"{prefix} {record.levelname.lower()}: {record.getMessage()}"


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced mlkit logger."""
    return logging.getLogger(name if name.startswith("mlkit") else f"mlkit.{name}")


def configure_logging(level: int | str = logging.INFO, json_format: bool = False) -> None:
    """Configure the root ``mlkit`` logger (idempotent)."""
    logger = logging.getLogger("mlkit")
    logger.setLevel(level)
    for h in list(logger.handlers):
        if getattr(h, "_mlkit_handler", False):
            logger.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_format else _TextFormatter())
    handler._mlkit_handler = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.propagate = False
