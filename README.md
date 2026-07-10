# Churn ML System

A modular machine learning system that predicts customer churn, served through a production-ready FastAPI web service. The project follows a clean, component-based architecture that separates data ingestion, validation, transformation, training, evaluation, and inference into independently testable building blocks.

## Overview

The system trains a logistic regression model on customer subscription data and exposes real-time churn predictions over a REST API. It is designed to be reproducible and easy to extend — swap the model, add preprocessing steps, or retrain on new data without touching the serving layer.

## Features

- **Modular pipeline** — distinct components for ingestion, validation, transformation, training, and evaluation.
- **Config-driven** — paths and model selection live in a single `config/config.yaml`.
- **FastAPI service** — interactive OpenAPI docs and a `/predict` endpoint for live inference.
- **Reproducible artifacts** — trained model, preprocessor, and feature schema are serialized to `artifacts/`.
- **Lightweight** — pure scikit-learn + pandas stack, no heavyweight orchestration required.

## Project Structure

```
churn-ml-system/
├── app.py                      # FastAPI application exposing /predict
├── config/
│   └── config.yaml             # Data, artifact, and model configuration
├── src/
│   ├── components/             # Core ML components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluator.py
│   ├── pipeline/               # Orchestration layer
│   │   ├── train_pipeline.py   # End-to-end training flow
│   │   └── predict_pipeline.py # Loads artifacts and scores new data
│   └── utils/                  # Shared helpers and metrics
├── data/
│   ├── raw/churn.csv           # Source dataset
│   └── processed/              # Intermediate data (generated)
├── artifacts/                  # Trained model.pkl, preprocessor.pkl, feature_names.pkl
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.10 or newer
- `pip` and a virtual environment tool

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd churn-ml-system

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train the model

Run the training pipeline to ingest the raw data, validate it, fit the preprocessor, train the model, and persist artifacts:

```bash
python -m src.pipeline.train_pipeline
```

This writes the following to `artifacts/`:

| Artifact            | Description                                      |
| ------------------- | ------------------------------------------------ |
| `model.pkl`         | Trained logistic regression classifier          |
| `preprocessor.pkl`  | Fitted `ColumnTransformer` (scaling + encoding)  |
| `feature_names.pkl` | Ordered list of expected input features          |

### 2. Serve predictions

Start the FastAPI server with Uvicorn:

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://127.0.0.1:8000`.

- **Interactive docs:** `http://127.0.0.1:8000/docs`
- **OpenAPI schema:** `http://127.0.0.1:8000/openapi.json`

### API Reference

#### `GET /`

Health check. Returns a message indicating the service is running.

```json
{ "message": "Churn API is running. Use POST /predict" }
```

#### `POST /predict`

Predicts churn for a single customer.

**Request body**

```json
{
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
  "TotalCharges": "653.4"
}
```

**Response**

```json
{ "Churn": "No" }
```

**Example with curl**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.7,
    "TotalCharges": "151.65"
  }'
```

## How It Works

1. **Data Ingestion** loads the raw CSV defined in `config.yaml`.
2. **Data Validation** checks the schema and data quality before training.
3. **Data Transformation** drops identifiers, encodes the target (`Yes` → `1`, `No` → `0`), and builds a `ColumnTransformer` that standardizes numeric features and one-hot encodes categorical ones.
4. **Model Training** fits a `LogisticRegression` classifier.
5. **Model Evaluation** reports performance metrics on the training data.
6. **Prediction** loads the serialized artifacts and applies the same preprocessing to new records for consistent inference.

## Configuration

All paths and the active model are defined in `config/config.yaml`:

```yaml
data:
  raw_data_path: data/raw/churn.csv

artifacts:
  model_path: artifacts/model.pkl
  preprocessor_path: artifacts/preprocessor.pkl

model:
  name: logistic_regression
```

## Tech Stack

- **Python** — core language
- **FastAPI / Uvicorn** — REST API and ASGI server
- **scikit-learn** — preprocessing and modeling
- **pandas / numpy** — data handling
- **pydantic** — request validation
- **joblib** — artifact serialization

## License

This project is licensed under the terms in the `LICENSE` file.
