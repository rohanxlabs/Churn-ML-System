from __future__ import annotations

import pandas as pd
import pytest

from src.components.data_validation import DataValidation


def test_data_validation_reports_blank_total_charges(trained_artifact_dir):
    validator = DataValidation()
    df = pd.read_csv("data/raw/churn.csv")

    report = validator.validate(df)

    assert report["status"] == "passed"
    assert report["blank_total_charges"] == 11


def test_data_validation_rejects_missing_target_column():
    validator = DataValidation()
    df = pd.read_csv("data/raw/churn.csv").drop(columns=["Churn"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validator.validate(df)


def test_data_validation_rejects_invalid_binary_values():
    validator = DataValidation()
    df = pd.read_csv("data/raw/churn.csv")
    df.loc[0, "SeniorCitizen"] = 2

    with pytest.raises(ValueError, match="SeniorCitizen must be binary 0/1"):
        validator.validate(df)


def test_data_validation_rejects_non_numeric_total_charges():
    validator = DataValidation()
    df = pd.read_csv("data/raw/churn.csv")
    df.loc[0, "TotalCharges"] = "not-a-number"

    with pytest.raises(ValueError, match="TotalCharges contains non-numeric values"):
        validator.validate(df)
