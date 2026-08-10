from __future__ import annotations

import joblib 
import numpy as np
import pandas as pd 

from src.components.data_transformation import prepare_feature_frame
from src.utils.config import Settings, load_settings
from src.utils.logging import get_logger
from src.utils.metrics import risk_level


logger = get_logger(__name__)


class PredictPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.model = None
        self.preprocessor = None
        self.feature_names = None

    def _ensure_loaded(self) -> None:
        if self.model is not None and self.preprocessor is not None and self.feature_names is not None:
            return

        artifact_paths = [
            self.settings.model_path,
            self.settings.preprocessor_path,
            self.settings.feature_names_path,
        ]
        missing_artifacts = [str(path) for path in artifact_paths if not path.exists()]
        if missing_artifacts:
            raise FileNotFoundError(
                "Required model artifacts are missing. Run the training pipeline first. "
                f"Missing: {missing_artifacts}"
            )

        logger.info("Loading model artifacts from %s", self.settings.artifacts_dir)
        self.model = joblib.load(self.settings.model_path)
        self.preprocessor = joblib.load(self.settings.preprocessor_path)
        self.feature_names = joblib.load(self.settings.feature_names_path)

    def is_ready(self) -> bool:
        return (
            self.settings.model_path.exists()
            and self.settings.preprocessor_path.exists()
            and self.settings.feature_names_path.exists()
        )

    def _prepare_transformed_input(self, data: pd.DataFrame):
        logger.info("Loading input")
        prepared = prepare_feature_frame(data)

        missing = [col for col in self.feature_names if col not in prepared.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        prepared = prepared[self.feature_names]
        logger.info("Running preprocessing")
        return prepared, self.preprocessor.transform(prepared)

    def _top_drivers(self, transformed_features, probability: float) -> list[dict]:
        if not hasattr(self.model, "coef_") or not hasattr(self.preprocessor, "get_feature_names_out"):
            return []

        feature_names = self.preprocessor.get_feature_names_out()
        dense_features = (
            transformed_features.toarray().ravel()
            if hasattr(transformed_features, "toarray")
            else np.asarray(transformed_features).ravel()
        )
        contributions = dense_features * self.model.coef_[0]
        ranked_indexes = np.argsort(np.abs(contributions))[::-1]

        drivers = []
        for index in ranked_indexes:
            contribution = float(contributions[index])
            if contribution == 0:
                continue
            drivers.append(
                {
                    "feature": str(feature_names[index]),
                    "contribution": round(contribution, 4),
                    "direction": "increase" if contribution > 0 else "decrease",
                }
            )
            if len(drivers) == 3:
                break

        return drivers

    def predict(self, data: pd.DataFrame) -> list[dict]:
        self._ensure_loaded()
        prepared, transformed = self._prepare_transformed_input(data)

        logger.info("Generating prediction")
        predictions = self.model.predict(transformed)
        probabilities = self.model.predict_proba(transformed)[:, 1]

        results = []
        for row_index, prediction in enumerate(predictions):
            probability = float(probabilities[row_index])
            results.append(
                {
                    "churn_prediction": "Yes" if int(prediction) == 1 else "No",
                    "churn_probability": round(probability, 4),
                    "risk_level": risk_level(probability),
                    "top_drivers": self._top_drivers(transformed[row_index], probability),
                    "input_features": prepared.iloc[row_index].to_dict(),
                }
            )

        logger.info("Prediction completed")
        return results
        
