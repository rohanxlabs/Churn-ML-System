from sklearn.linear_model import LogisticRegression
import joblib

from src.utils.config import Settings, load_settings
from src.utils.logging import get_logger


logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()

    def train(self, X, y):
        model = LogisticRegression(
            max_iter=1000,
            class_weight=self.settings.class_weight,
            solver="liblinear",
            random_state=self.settings.random_state,
        )
        model.fit(X, y)
        joblib.dump(model, self.settings.model_path)
        logger.info("Saved model artifact to %s", self.settings.model_path)
        return model
