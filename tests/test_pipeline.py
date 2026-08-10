from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from src.pipeline.predict_pipeline import PredictPipeline


def test_training_generates_expected_artifacts(trained_artifact_dir):
    expected_files = [
        trained_artifact_dir / "model.pkl",
        trained_artifact_dir / "preprocessor.pkl",
        trained_artifact_dir / "feature_names.pkl",
        trained_artifact_dir / "metrics.json",
        trained_artifact_dir / "validation_report.json",
    ]

    for path in expected_files:
        assert path.exists(), f"Missing artifact: {path}"

    metrics = json.loads((trained_artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert set(metrics).issuperset({"accuracy", "precision", "recall", "f1", "roc_auc"})


def test_prediction_pipeline_returns_probability_and_drivers(trained_artifact_dir):
    pipeline = PredictPipeline()
    row = pd.read_csv("data/raw/churn.csv").drop(columns=["Churn"]).head(1)

    result = pipeline.predict(row)[0]

    assert result["churn_prediction"] in {"Yes", "No"}
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_level"] in {"low", "medium", "high"}
    assert len(result["top_drivers"]) >= 1


def test_prediction_pipeline_rejects_missing_features(trained_artifact_dir):
    pipeline = PredictPipeline()
    row = pd.read_csv("data/raw/churn.csv").drop(columns=["Churn", "tenure"]).head(1)

    with pytest.raises(ValueError, match="Missing required features"):
        pipeline.predict(row)


def test_prediction_pipeline_reports_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("CHURN_ARTIFACT_DIR", str(tmp_path))
    pipeline = PredictPipeline()

    with pytest.raises(FileNotFoundError, match="Run the training pipeline first"):
        pipeline.predict(pd.read_csv("data/raw/churn.csv").drop(columns=["Churn"]).head(1))
