from __future__ import annotations

from functools import lru_cache
from typing import Literal

import pandas as pd 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.pipeline.predict_pipeline import PredictPipeline
from src.utils.config import Settings, load_settings
from src.utils.logging import configure_logging, get_logger


configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Churn ML System API",
    description="API for customer churn inference using the trained churn model artifact.",
    version="1.0.0",
)


class TopDriver(BaseModel):
    feature: str
    contribution: float
    direction: Literal["increase", "decrease"]


class PredictionResponse(BaseModel):
    churn_prediction: Literal["Yes", "No"]
    churn_probability: float
    risk_level: Literal["low", "medium", "high"]
    top_drivers: list[TopDriver]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    artifacts_dir: str


class ErrorResponse(BaseModel):
    detail: str


@lru_cache
def get_settings() -> Settings:
    settings = load_settings()
    configure_logging(settings.log_level)
    return settings


@lru_cache
def get_pipeline() -> PredictPipeline:
    return PredictPipeline(settings=get_settings())


class ChurnInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 653.4,
            }
        }
    )

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.get("/", response_model=dict)
def home():
    return {"message": "Churn ML System API is running. Visit /docs for interactive API docs."}


@app.get("/health", response_model=HealthResponse)
def health():
    pipeline = get_pipeline()
    settings = get_settings()
    return HealthResponse(
        status="ok" if pipeline.is_ready() else "degraded",
        model_loaded=pipeline.is_ready(),
        artifacts_dir=str(settings.artifacts_dir),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def predict_churn(data: ChurnInput):
    df = pd.DataFrame([data.model_dump()])
    pipeline = get_pipeline()

    try:
        return PredictionResponse(**pipeline.predict(df)[0])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

