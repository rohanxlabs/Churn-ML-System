from __future__ import annotations

import pandas as pd

from src.utils.common import write_json
from src.utils.config import Settings, load_settings
from src.utils.logging import get_logger


logger = get_logger(__name__)

REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

YES_NO_COLUMNS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]


class DataValidation:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()

    def validate(self, df: pd.DataFrame) -> dict:
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        invalid_target_values = sorted(
            set(df["Churn"].dropna().astype(str).unique()) - {"Yes", "No"}
        )
        if invalid_target_values:
            raise ValueError(f"Unexpected target values found: {invalid_target_values}")

        invalid_yes_no_values = {
            column: sorted(set(df[column].dropna().astype(str).unique()) - {"Yes", "No"})
            for column in YES_NO_COLUMNS
        }
        invalid_yes_no_values = {
            column: values for column, values in invalid_yes_no_values.items() if values
        }
        if invalid_yes_no_values:
            raise ValueError(f"Unexpected categorical values found: {invalid_yes_no_values}")

        null_counts = {column: int(count) for column, count in df.isnull().sum().items() if count > 0}
        if null_counts:
            raise ValueError(f"Null values found in dataset: {null_counts}")

        blank_total_charges = int(df["TotalCharges"].astype(str).str.strip().eq("").sum())
        total_charges_numeric = pd.to_numeric(
            df["TotalCharges"].replace(r"^\s*$", "0", regex=True),
            errors="coerce",
        )
        if total_charges_numeric.isna().any():
            raise ValueError("TotalCharges contains non-numeric values.")

        monthly_charges_numeric = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        if monthly_charges_numeric.isna().any():
            raise ValueError("MonthlyCharges contains non-numeric values.")
        if (monthly_charges_numeric < 0).any():
            raise ValueError("MonthlyCharges contains negative values.")

        tenure_numeric = pd.to_numeric(df["tenure"], errors="coerce")
        if tenure_numeric.isna().any():
            raise ValueError("tenure contains non-numeric values.")
        if (tenure_numeric < 0).any():
            raise ValueError("tenure contains negative values.")

        senior_citizen_numeric = pd.to_numeric(df["SeniorCitizen"], errors="coerce")
        if senior_citizen_numeric.isna().any():
            raise ValueError("SeniorCitizen contains non-numeric values.")
        senior_citizen_values = sorted(
            set(senior_citizen_numeric.astype(int).unique()) - {0, 1}
        )
        if senior_citizen_values:
            raise ValueError(
                f"SeniorCitizen must be binary 0/1. Found: {senior_citizen_values}"
            )
        report = {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "missing_columns": missing_columns,
            "null_counts": null_counts,
            "blank_total_charges": blank_total_charges,
            "valid_target_values": ["Yes", "No"],
            "status": "passed",
        }

        write_json(self.settings.validation_report_path, report)
        if blank_total_charges:
            logger.warning(
                "Detected %s blank TotalCharges values; they will be coerced to 0.0 during transformation",
                blank_total_charges,
            )

        logger.info("Dataset validation completed")
        return report
