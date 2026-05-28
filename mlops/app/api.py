"""
Servicio de inferencia — Telco Churn Prediction
FastAPI + joblib pipeline (XGBoost)
"""

from __future__ import annotations

import os
import time
import logging
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


MODEL_PATH = os.getenv("MODEL_PATH", "app/model.joblib")
CHURN_THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.5"))


model_store: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Cargando modelo desde %s …", MODEL_PATH)

    try:
  
        if "pipeline" not in model_store:
            model_store["pipeline"] = joblib.load(MODEL_PATH)

        logger.info("Modelo cargado correctamente.")

    except FileNotFoundError:
        logger.warning(
            "Modelo no encontrado en %s. El servicio responderá 503.",
            MODEL_PATH
        )

    except Exception as exc:
        logger.error("Error inesperado al cargar el modelo: %s", exc)

    yield

    model_store.clear()
    logger.info("Modelo descargado. Servicio apagado.")


# App 
app = FastAPI(
    title="Telco Churn Prediction API",
    description="Servicio de inferencia para predicción de churn.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Middleware de latencia
@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    start = time.perf_counter()

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"

    return response


# Schemas
class CustomerFeatures(BaseModel):
    gender: str = Field(..., examples=["Male", "Female"])
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: float = Field(..., ge=0)

    PhoneService: str
    MultipleLines: str
    InternetService: str

    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str

    StreamingTV: str
    StreamingMovies: str

    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        if v not in {"Male", "Female"}:
            raise ValueError("gender debe ser Male o Female")
        return v

    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, v):
        valid = {"Month-to-month", "One year", "Two year"}

        if v not in valid:
            raise ValueError(f"Contract debe ser uno de {valid}")

        return v


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(
        ...,
        min_length=1,
        max_length=1000,
    )


class PredictionResult(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: str


class SingleResponse(BaseModel):
    prediction: PredictionResult
    model_version: str
    threshold_used: float

    model_config = {"protected_namespaces": ()}


class BatchResponse(BaseModel):
    predictions: list[PredictionResult]
    total_customers: int
    predicted_churners: int
    churn_rate_predicted: float
    model_version: str

    model_config = {"protected_namespaces": ()}


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    threshold: float

    model_config = {"protected_namespaces": ()}


# Helpers 
def _risk_level(prob: float) -> str:
    if prob >= 0.70:
        return "ALTO"

    if prob >= 0.40:
        return "MEDIO"

    return "BAJO"


def _require_model():
    if "pipeline" not in model_store:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible."
        )


# Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded="pipeline" in model_store,
        threshold=CHURN_THRESHOLD,
    )


@app.post("/predict", response_model=SingleResponse, tags=["Inferencia"])
def predict(customer: CustomerFeatures):

    _require_model()

    df = pd.DataFrame([customer.model_dump()])

    try:
        prob = float(
            model_store["pipeline"]
            .predict_proba(df)[0, 1]
        )

    except Exception as exc:
        logger.exception("Error en inferencia: %s", exc)

        raise HTTPException(
            status_code=500,
            detail=f"Error en inferencia: {exc}"
        )

    prediction = PredictionResult(
        churn_probability=round(prob, 4),
        churn_prediction=prob >= CHURN_THRESHOLD,
        risk_level=_risk_level(prob),
    )

    logger.info(
        "predict | prob=%.4f | risk=%s | churn=%s",
        prob,
        prediction.risk_level,
        prediction.churn_prediction,
    )

    return SingleResponse(
        prediction=prediction,
        model_version="xgboost-v1.0",
        threshold_used=CHURN_THRESHOLD,
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["Inferencia"])
def predict_batch(request: BatchRequest):

    _require_model()

    df = pd.DataFrame([
        c.model_dump()
        for c in request.customers
    ])

    try:
        probs = (
            model_store["pipeline"]
            .predict_proba(df)[:, 1]
        )

    except Exception as exc:
        logger.exception("Error en inferencia batch: %s", exc)

        raise HTTPException(
            status_code=500,
            detail=f"Error en inferencia: {exc}"
        )

    predictions = [
        PredictionResult(
            churn_probability=round(float(p), 4),
            churn_prediction=float(p) >= CHURN_THRESHOLD,
            risk_level=_risk_level(float(p)),
        )
        for p in probs
    ]

    churners = sum(
        1 for p in predictions
        if p.churn_prediction
    )

    logger.info(
        "predict_batch | n=%d | churners=%d",
        len(predictions),
        churners,
    )

    return BatchResponse(
        predictions=predictions,
        total_customers=len(predictions),
        predicted_churners=churners,
        churn_rate_predicted=round(
            churners / len(predictions),
            4
        ),
        model_version="xgboost-v1.0",
    )