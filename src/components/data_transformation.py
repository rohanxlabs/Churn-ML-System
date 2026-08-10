from __future__ import annotations

import joblib 
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.utils.config import Settings, load_settings
from src.utils.logging import get_logger


logger = get_logger(__name__)


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()

    if "customerID" in prepared.columns:
        prepared = prepared.drop(columns=["customerID"])

    if "TotalCharges" in prepared.columns:
        prepared["TotalCharges"] = pd.to_numeric(prepared["TotalCharges"], errors="coerce").fillna(0.0)

    return prepared


class DataTransformation:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.preprocessor: ColumnTransformer | None = None
        self.feature_names: list[str] | None = None

    def fit_transform(self, df: pd.DataFrame, target_col: str):
        prepared = prepare_feature_frame(df)
        X = prepared.drop(columns=[target_col])
        y = prepared[target_col].map({"Yes": 1, "No": 0})

        self.feature_names = X.columns.tolist()
        joblib.dump(self.feature_names, self.settings.feature_names_path)

        num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        cat_cols = [column for column in X.columns if column not in num_cols]

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                    cat_cols,
                ),
            ]
        )

        X_transformed = self.preprocessor.fit_transform(X)
        joblib.dump(self.preprocessor, self.settings.preprocessor_path)
        logger.info("Saved preprocessor and feature schema to %s", self.settings.artifacts_dir)
        return X_transformed, y

    def transform(self, df: pd.DataFrame, target_col: str | None = None):
        if self.preprocessor is None:
            self.preprocessor = joblib.load(self.settings.preprocessor_path)
        if self.feature_names is None:
            self.feature_names = joblib.load(self.settings.feature_names_path)

        prepared = prepare_feature_frame(df)
        y = None

        if target_col and target_col in prepared.columns:
            y = prepared[target_col].map({"Yes": 1, "No": 0})
            prepared = prepared.drop(columns=[target_col])

        missing_columns = [column for column in self.feature_names if column not in prepared.columns]
        if missing_columns:
            raise ValueError(f"Missing required features: {missing_columns}")

        prepared = prepared[self.feature_names]
        X_transformed = self.preprocessor.transform(prepared)

        if y is None:
            return X_transformed
        return X_transformed, y
