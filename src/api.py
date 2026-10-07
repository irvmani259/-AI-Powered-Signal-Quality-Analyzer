"""
api.py
------
FastAPI application exposing all signal-quality-analyzer endpoints.

Endpoints
---------
POST /upload          Upload CSV or JSON file; triggers ingestion + detection
GET  /anomalies       Return anomaly summary (optionally filtered by node/severity)
GET  /insights        Return AI insights (Granite or mock)
GET  /forecast        Return forecasts + failure risk ranking
GET  /health          Service liveness + dataset summary
GET  /nodes           List available node IDs
GET  /raw             Return raw (cleaned) data for a given node

State
-----
A simple in-memory store (AppState) holds the last ingested dataset and all
derived results.  For production, replace with a database or Cloud Object Storage.
"""

from __future__ import annotations

import io
import logging
import sys
import os
from pathlib import Path
from typing import Any, Optional

# Make src/ importable when running directly
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ingestion import ingest, generate_synthetic
from detection import detect, DetectionResult
from forecasting import forecast, ForecastResult
from insights import generate_insights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI-Powered Signal Quality Analyzer",
    description=(
        "Analyzes telecom signal metrics (SNR, BER, latency), detects anomalies, "
        "forecasts failures, and generates AI-driven insights via IBM Granite."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

class AppState:
    df: Optional[pd.DataFrame] = None
    detection: Optional[DetectionResult] = None
    forecast_result: Optional[ForecastResult] = None
    insights_cache: Optional[dict[str, Any]] = None
    source_filename: str = ""

state = AppState()


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    dataset_loaded: bool
    n_rows: int
    n_nodes: int
    source_filename: str


class UploadResponse(BaseModel):
    status: str
    rows_ingested: int
    nodes: list[str]
    anomaly_count: int
    critical_count: int
    message: str


# ---------------------------------------------------------------------------
# Helper: ensure data is loaded
# ---------------------------------------------------------------------------

def _require_data() -> None:
    if state.df is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "No dataset loaded. POST to /upload first, "
                "or call /upload with use_synthetic=true."
            ),
        )


def _to_json_safe(obj: Any) -> Any:
    """Recursively make an object JSON-serialisable."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(i) for i in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

@app.post("/upload", response_model=UploadResponse)
async def upload_data(
    file: Optional[UploadFile] = File(default=None),
    use_synthetic: bool = Query(default=False, description="Generate and use synthetic data"),
    n_nodes: int = Query(default=5, ge=1, le=20),
    hours: int = Query(default=48, ge=6, le=720),
):
    """
    Upload a CSV or JSON file OR request synthetic data generation.
    Triggers the full ingestion → detection pipeline automatically.
    """
    try:
        if use_synthetic or file is None:
            logger.info("Generating synthetic dataset: %d nodes, %d hours", n_nodes, hours)
            raw_df = generate_synthetic(n_nodes=n_nodes, hours=hours)
            state.source_filename = f"synthetic_{n_nodes}nodes_{hours}h"
        else:
            content = await file.read()
            fmt = "json" if file.filename and file.filename.endswith(".json") else "csv"
            raw_df = ingest(content, fmt=fmt)
            state.source_filename = file.filename or "uploaded_file"

        state.df = ingest(raw_df) if use_synthetic else raw_df
        state.detection = detect(state.df)
        state.forecast_result = None   # lazy — computed on first /forecast call
        state.insights_cache = None    # lazy — computed on first /insights call

        nodes = sorted(state.df["node_id"].unique().tolist())
        anomaly_count = len(state.detection.anomalies)
        critical_count = int((state.detection.anomalies["severity"] == "Critical").sum()) if not state.detection.anomalies.empty else 0

        return UploadResponse(
            status="ok",
            rows_ingested=len(state.df),
            nodes=nodes,
            anomaly_count=anomaly_count,
            critical_count=critical_count,
            message=f"Successfully processed {len(state.df):,} rows across {len(nodes)} nodes.",
        )

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /anomalies
# ---------------------------------------------------------------------------

@app.get("/anomalies")
def get_anomalies(
    node_id: Optional[str] = Query(default=None),
    severity: Optional[str] = Query(default=None, description="Low|Medium|High|Critical"),
    limit: int = Query(default=500, ge=1, le=5000),
):
    """Return detected anomalies, optionally filtered by node and/or severity."""
    _require_data()
    anomalies = state.detection.anomalies.copy()

    if node_id:
        anomalies = anomalies[anomalies["node_id"] == node_id]
    if severity:
        anomalies = anomalies[anomalies["severity"].str.lower() == severity.lower()]

    anomalies = anomalies.head(limit)

    return JSONResponse(content={
        "total": len(anomalies),
        "interference_events": _to_json_safe(state.detection.interference_events),
        "degradation_events": _to_json_safe(state.detection.degradation_events),
        "node_health": _to_json_safe(state.detection.node_health),
        "anomalies": _to_json_safe(state.detection.summary[:limit]),
    })


# ---------------------------------------------------------------------------
# GET /insights
# ---------------------------------------------------------------------------

@app.get("/insights")
def get_insights(refresh: bool = Query(default=False)):
    """
    Return AI-generated insights from IBM Granite (or mock fallback).
    Results are cached after the first call; pass refresh=true to recompute.
    """
    _require_data()

    if state.insights_cache is None or refresh:
        state.insights_cache = generate_insights(
            anomaly_summary=state.detection.summary,
            interference_events=state.detection.interference_events,
            degradation_events=state.detection.degradation_events,
            node_health=state.detection.node_health,
        )

    return JSONResponse(content=_to_json_safe(state.insights_cache))


# ---------------------------------------------------------------------------
# GET /forecast
# ---------------------------------------------------------------------------

@app.get("/forecast")
def get_forecast(
    horizon_hours: int = Query(default=12, ge=1, le=168),
    node_id: Optional[str] = Query(default=None),
):
    """
    Return forecasts for SNR, BER, and latency plus failure risk ranking.
    Forecasting is computed lazily and cached.
    """
    _require_data()

    # Re-run if horizon changed or not yet computed
    if state.forecast_result is None:
        logger.info("Running forecast pipeline (horizon=%dh)", horizon_hours)
        state.forecast_result = forecast(state.df, horizon_hours=horizon_hours)

    fr = state.forecast_result

    # Serialize forecasts
    serialized: dict[str, Any] = {}
    for node, metrics in fr.forecasts.items():
        if node_id and node != node_id:
            continue
        serialized[node] = {}
        for metric, fdf in metrics.items():
            if fdf is not None and not fdf.empty:
                serialized[node][metric] = _to_json_safe(fdf)

    return JSONResponse(content={
        "forecasts": serialized,
        "accuracy": _to_json_safe(fr.accuracy),
        "failure_probability": _to_json_safe(fr.failure_probability),
        "time_to_failure": _to_json_safe(fr.time_to_failure),
        "risk_ranking": _to_json_safe(fr.risk_ranking),
    })


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Service liveness and dataset summary."""
    return HealthResponse(
        status="ok",
        dataset_loaded=state.df is not None,
        n_rows=len(state.df) if state.df is not None else 0,
        n_nodes=state.df["node_id"].nunique() if state.df is not None else 0,
        source_filename=state.source_filename,
    )


# ---------------------------------------------------------------------------
# GET /nodes
# ---------------------------------------------------------------------------

@app.get("/nodes")
def list_nodes():
    """List all node IDs in the loaded dataset."""
    _require_data()
    nodes = sorted(state.df["node_id"].unique().tolist())
    health_map: dict[str, Any] = {}
    if state.detection and not state.detection.node_health.empty:
        for _, row in state.detection.node_health.iterrows():
            health_map[row["node_id"]] = {
                "health_score": row["health_score"],
                "mean_snr": row["mean_snr"],
                "mean_latency": row["mean_latency"],
            }
    return {"nodes": nodes, "health": health_map}


# ---------------------------------------------------------------------------
# GET /raw
# ---------------------------------------------------------------------------

@app.get("/raw")
def get_raw_data(
    node_id: Optional[str] = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=10000),
):
    """Return the cleaned time-series data for charting."""
    _require_data()
    df = state.df.copy()
    if node_id:
        df = df[df["node_id"] == node_id]
    df = df.tail(limit)
    return JSONResponse(content=_to_json_safe(df[
        ["timestamp", "node_id", "snr", "ber", "latency"]
        + (["packet_loss"] if "packet_loss" in df.columns else [])
    ]))


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
