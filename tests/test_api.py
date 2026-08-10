from __future__ import annotations

import importlib

import pandas as pd
from fastapi.testclient import TestClient


def _load_app_module():
    import app as app_module

    importlib.reload(app_module)
    app_module.get_settings.cache_clear()
    app_module.get_pipeline.cache_clear()
    return app_module


def test_health_endpoint_reports_model_ready(trained_artifact_dir):
    app_module = _load_app_module()

    with TestClient(app_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict_endpoint_returns_prediction(trained_artifact_dir):
    app_module = _load_app_module()
    payload = (
        pd.read_csv("data/raw/churn.csv")
        .drop(columns=["customerID", "Churn"])
        .iloc[0]
        .to_dict()
    )
    payload["TotalCharges"] = float(payload["TotalCharges"])

    with TestClient(app_module.app) as client:
        response = client.post("/predict", json=payload)
        second_response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert second_response.status_code == 200
    body = response.json()
    assert set(body).issuperset({"churn_prediction", "churn_probability", "risk_level", "top_drivers"})


def test_predict_endpoint_rejects_missing_fields(trained_artifact_dir):
    app_module = _load_app_module()

    with TestClient(app_module.app) as client:
        response = client.post("/predict", json={"gender": "Female"})

    assert response.status_code == 422


def test_predict_endpoint_rejects_invalid_values(trained_artifact_dir):
    app_module = _load_app_module()
    payload = (
        pd.read_csv("data/raw/churn.csv")
        .drop(columns=["customerID", "Churn"])
        .iloc[0]
        .to_dict()
    )
    payload["gender"] = "Unknown"
    payload["TotalCharges"] = float(payload["TotalCharges"])

    with TestClient(app_module.app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_endpoint_rejects_invalid_data_types(trained_artifact_dir):
    app_module = _load_app_module()
    payload = (
        pd.read_csv("data/raw/churn.csv")
        .drop(columns=["customerID", "Churn"])
        .iloc[0]
        .to_dict()
    )
    payload["TotalCharges"] = "not-a-number"

    with TestClient(app_module.app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_openapi_schema_exposes_prediction_models(trained_artifact_dir):
    app_module = _load_app_module()

    with TestClient(app_module.app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "Churn ML System API"
    assert "ChurnInput" in schema["components"]["schemas"]
    assert "PredictionResponse" in schema["components"]["schemas"]


def test_predict_endpoint_returns_503_when_model_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("CHURN_ARTIFACT_DIR", str(tmp_path))
    app_module = _load_app_module()

    with TestClient(app_module.app) as client:
        health_response = client.get("/health")
        predict_response = client.post(
            "/predict",
            json={
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
            },
        )

    assert health_response.status_code == 200
    assert health_response.json()["status"] == "degraded"
    assert predict_response.status_code == 503
