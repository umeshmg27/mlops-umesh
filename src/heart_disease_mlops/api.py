from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from heart_disease_mlops.config import MODEL_METADATA_PATH, MODEL_PATH
from heart_disease_mlops.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PatientRecord,
    PredictionResponse,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("heart_disease_api")

REQUEST_COUNT = Counter(
    "heart_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)
PREDICTION_COUNT = Counter(
    "heart_predictions_total",
    "Total predictions by predicted class",
    ["prediction"],
)
PREDICTION_LATENCY = Histogram(
    "heart_prediction_latency_seconds",
    "Prediction latency in seconds",
)

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="FastAPI service for the UCI Heart Disease MLOps assignment.",
    version="0.1.0",
)

MODEL_BUNDLE: dict[str, Any] | None = None


@app.middleware("http")
async def metrics_and_logging(request: Request, call_next):
    start = time.perf_counter()
    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        elapsed = time.perf_counter() - start
        endpoint = request.url.path
        REQUEST_COUNT.labels(request.method, endpoint, status).inc()
        logger.info(
            "request method=%s path=%s status=%s duration=%.4f",
            request.method,
            endpoint,
            status,
            elapsed,
        )


def _model_paths() -> tuple[Path, Path]:
    model_path = Path(os.getenv("MODEL_PATH", str(MODEL_PATH)))
    metadata_path = Path(os.getenv("MODEL_METADATA_PATH", str(MODEL_METADATA_PATH)))
    return model_path, metadata_path


def load_model_bundle(force: bool = False) -> dict[str, Any]:
    global MODEL_BUNDLE
    if MODEL_BUNDLE is not None and not force:
        return MODEL_BUNDLE

    model_path, metadata_path = _model_paths()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    MODEL_BUNDLE = {
        "model": joblib.load(model_path),
        "metadata": metadata,
    }
    logger.info("loaded model from %s", model_path)
    return MODEL_BUNDLE


def _predict(records: list[PatientRecord]) -> list[PredictionResponse]:
    try:
        bundle = load_model_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    model = bundle["model"]
    metadata = bundle.get("metadata", {})
    model_name = metadata.get("model_name", "unknown")

    frame = pd.DataFrame([record.model_dump() for record in records])
    predictions = model.predict(frame)
    probabilities = model.predict_proba(frame)
    positive_index = list(model.classes_).index(1)

    responses: list[PredictionResponse] = []
    for prediction, probability_row in zip(predictions, probabilities, strict=True):
        risk_probability = float(probability_row[positive_index])
        confidence = float(max(probability_row))
        label = "Heart disease risk" if int(prediction) == 1 else "No heart disease risk"
        PREDICTION_COUNT.labels(str(int(prediction))).inc()
        responses.append(
            PredictionResponse(
                prediction=int(prediction),
                label=label,
                risk_probability=risk_probability,
                confidence=confidence,
                model_name=model_name,
            )
        )
    return responses


@app.get("/health")
def health() -> dict[str, object]:
    model_path, metadata_path = _model_paths()
    return {
        "status": "ok",
        "model_available": model_path.exists(),
        "model_path": str(model_path),
        "metadata_available": metadata_path.exists(),
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    try:
        bundle = load_model_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return bundle.get("metadata", {})


@app.post("/predict", response_model=PredictionResponse)
def predict(record: PatientRecord) -> PredictionResponse:
    start = time.perf_counter()
    try:
        return _predict([record])[0]
    finally:
        PREDICTION_LATENCY.observe(time.perf_counter() - start)


@app.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    if not payload.records:
        raise HTTPException(status_code=400, detail="records must not be empty")
    start = time.perf_counter()
    try:
        return BatchPredictionResponse(predictions=_predict(payload.records))
    finally:
        PREDICTION_LATENCY.observe(time.perf_counter() - start)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

