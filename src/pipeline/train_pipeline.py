from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_evaluator import ModelEvaluator
from src.components.model_trainer import ModelTrainer

def run_training():
    ingestion = DataIngestion("config/config.yaml")
    df = ingestion.ingest()

    DataValidation().validate(df)

    X,y = DataTransformation().transform(df,"Churn")

    model = ModelTrainer().train(X,y)

    ModelEvaluator().evaluate(model,X,y)

    print("train pipeline successfully")

if __name__ == "__main__":
    run_training()