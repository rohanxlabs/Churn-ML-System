from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.common import write_json
from src.utils.config import Settings, load_settings
from src.utils.logging import get_logger


logger = get_logger(__name__)


class ModelEvaluator:
    def __init__(self, settings: Settings | None = None, preprocessor=None):
        self.settings = settings or load_settings()
        self.preprocessor = preprocessor

    def evaluate(self, model, X, y):
        preds = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        metrics = {
            "accuracy": round(float(accuracy_score(y, preds)), 4),
            "precision": round(float(precision_score(y, preds)), 4),
            "recall": round(float(recall_score(y, preds)), 4),
            "f1": round(float(f1_score(y, preds)), 4),
            "roc_auc": round(float(roc_auc_score(y, probabilities)), 4),
            "confusion_matrix": confusion_matrix(y, preds).tolist(),
            "positive_rate": round(float(probabilities.mean()), 4),
            "evaluation_rows": int(len(y)),
        }

        if self.preprocessor is not None and hasattr(self.preprocessor, "get_feature_names_out"):
            feature_names = self.preprocessor.get_feature_names_out()
            coefficients = model.coef_[0]
            ranked_features = sorted(
                (
                    {"feature": name, "coefficient": round(float(coef), 4)}
                    for name, coef in zip(feature_names, coefficients)
                ),
                key=lambda item: abs(item["coefficient"]),
                reverse=True,
            )
            metrics["top_feature_importance"] = ranked_features[:10]

        write_json(self.settings.metrics_path, metrics)
        logger.info("Saved evaluation metrics to %s", self.settings.metrics_path)
        return metrics
