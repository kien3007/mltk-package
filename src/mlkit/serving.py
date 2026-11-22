"""HTTP model serving with FastAPI (``pip install 'mlkit[serve]'``).

Endpoints:
    GET  /health    liveness + bundle info
    GET  /metadata  config, schema and training metrics
    POST /predict   {"records": [{...}, ...], "threshold": 0.5} -> {"predictions": [...]}
    POST /monitor   {"records": [...]} -> drift report for the posted batch
"""

import contextlib
from pathlib import Path
from typing import Any

import pandas as pd

from mlkit.core.bundle import load_bundle
from mlkit.core.errors import ValidationError
from mlkit.monitoring import DriftMonitor
from mlkit.runner import model_output, predict_frame, validate_input


def create_app(bundle_path: str | Path) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise ImportError("Serving requires: pip install 'mlkit[serve]'") from exc

    pipeline, cfg, metadata = load_bundle(bundle_path)
    baseline = metadata.get("baseline") or {}
    monitor = DriftMonitor.from_dict(baseline) if baseline else None

    class PredictRequest(BaseModel):
        records: list[dict[str, Any]] = Field(..., min_length=1)
        threshold: float = Field(0.5, ge=0.0, le=1.0)

    class MonitorRequest(BaseModel):
        records: list[dict[str, Any]] = Field(..., min_length=1)

    app = FastAPI(
        title="mlkit model server", version=str(metadata.get("versions", {}).get("mlkit", ""))
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "task_type": cfg.task_type,
            "champion": metadata.get("metrics", {}).get("champion"),
        }

    @app.get("/metadata")
    def get_metadata() -> dict[str, Any]:
        return {
            "config": cfg.model_dump(mode="json"),
            "schema": metadata.get("schema", {}),
            "metrics": metadata.get("metrics", {}),
            "versions": metadata.get("versions", {}),
        }

    @app.post("/predict")
    def predict(req: PredictRequest) -> dict[str, Any]:
        df = pd.DataFrame.from_records(req.records)
        try:
            out = predict_frame(pipeline, cfg, metadata, df, threshold=req.threshold)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"predictions": out.to_dict(orient="records")}

    @app.post("/monitor")
    def monitor_batch(req: MonitorRequest) -> dict[str, Any]:
        if monitor is None:
            raise HTTPException(status_code=409, detail="Bundle has no drift baseline.")
        df = pd.DataFrame.from_records(req.records)
        y_pred = None
        with contextlib.suppress(ValidationError):
            y_pred = model_output(
                pipeline, validate_input(df, metadata.get("schema", {})), cfg.task_type
            )
        return monitor.monitor(df, y_pred=y_pred).to_dict()

    return app
