from __future__ import annotations

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_evaluator import ModelEvaluator
from src.components.model_trainer import ModelTrainer
from src.utils.config import load_settings
from src.utils.logging import configure_logging, get_logger
from sklearn.model_selection import train_test_split


logger = get_logger(__name__)


def run_training(config_path: str | None = None):
    settings = load_settings(config_path)
    configure_logging(settings.log_level)

    ingestion = DataIngestion(settings=settings)
    df = ingestion.ingest()

    validation_report = DataValidation(settings=settings).validate(df)

    train_df, test_df = train_test_split(
        df,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=df["Churn"],
    )
    logger.info(
        "Split dataset into %s train rows and %s test rows",
        len(train_df),
        len(test_df),
    )

    transformer = DataTransformation(settings=settings)
    X_train, y_train = transformer.fit_transform(train_df, "Churn")
    X_test, y_test = transformer.transform(test_df, "Churn")

    model = ModelTrainer(settings=settings).train(X_train, y_train)
    metrics = ModelEvaluator(
        settings=settings,
        preprocessor=transformer.preprocessor,
    ).evaluate(model, X_test, y_test)

    logger.info("Training pipeline completed successfully")
    return {
        "validation_report": validation_report,
        "metrics": metrics,
        "artifacts_dir": str(settings.artifacts_dir),
    }

if __name__ == "__main__":
    run_training()
