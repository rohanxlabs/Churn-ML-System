
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

## Deployment

This project is configured for [Render](https://render.com). A `render.yaml`
and `Procfile` are included so the service starts correctly.

> **Note:** FastAPI is an ASGI framework. It must be served with an ASGI worker
> (Uvicorn), not Gunicorn's default WSGI worker. The included config uses
> `gunicorn` with `-k uvicorn.workers.UvicornWorker` and binds to the
> `$PORT` environment variable that Render provides.

### Deploy with `render.yaml`

1. Push the repository to GitHub.
2. In Render, create a new **Blueprint** and connect the repo — the
   `render.yaml` is detected automatically.
3. The build command installs dependencies and the start command launches the
   ASGI server.

### Deploy manually (Web Service)

- **Build command:** `pip install -r requirements.txt`
- **Start command:**
  ```bash
  gunicorn app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
  ```

### Local equivalent

```bash
gunicorn app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
# or simply
python -m uvicorn app:app --host 0.0.0.0 --port 8000
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

Live link : https://churn-ml-system.onreder.com
App link : https://churn-monitor.lovable.app

📉 Customer Churn Prediction System

⚡ Predicting User Drop-Off with Machine Learning

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?size=22&duration=3000&color=FF3C38&center=true&vCenter=true&width=750&lines=Customer+Churn+Prediction;From+Data+→+Insights+→+Retention;Machine+Learning+for+Business+Impact" />
</p><p align="center">
  <img src="https://img.shields.io/badge/ML-Classification-blue">
  <img src="https://img.shields.io/badge/Problem-Churn Prediction-orange">
  <img src="https://img.shields.io/badge/API-FastAPI-green">
  <img src="https://img.shields.io/badge/Status-Production Ready-success">
</p>---

🎯 Problem Statement

Customer churn is one of the biggest challenges for businesses.

- Losing customers directly impacts revenue
- Acquiring new customers is more expensive than retaining existing ones

This project builds a machine learning system to predict customer churn, enabling proactive retention strategies.

Predictive churn systems typically rely on user behavior, transactions, and engagement patterns to identify at-risk users early.

---

💡 Business Impact

A churn prediction system helps:

- 📉 Reduce customer loss
- 🎯 Target high-risk users
- 💰 Improve retention strategies
- 📊 Drive data-driven decisions

---

🧠 ML Problem Formulation

- Type: Binary Classification
- Target: Churn (Yes / No)
- Input: Customer features (usage, behavior, demographics)

---

🏗️ System Architecture

Raw Data
   ↓
Data Preprocessing
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Evaluation
   ↓
Prediction API
   ↓
Business Action (Retention Strategy)

---

⚙️ Core Components

📥 Data Processing

- Handle missing values
- Encode categorical features
- Normalize data

📊 Feature Engineering

- Extract meaningful patterns
- Improve model performance

🤖 Model Training

- Train classification models
- Compare performance

📈 Evaluation

- Accuracy, Precision, Recall
- Model selection

🌐 API Deployment

- Serve predictions using FastAPI

---

🔄 Pipeline Workflow

1. Load dataset
2. Clean & preprocess data
3. Perform feature engineering
4. Train ML model
5. Evaluate performance
6. Deploy via API

---

🛠️ Tech Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,fastapi,git" />
</p>- Python
- Pandas / NumPy
- Scikit-learn
- FastAPI

---

📂 Project Structure

churn-ml-system/
│
├── data/                → Dataset
├── models/              → Trained models
├── notebooks/           → EDA & experiments
├── src/                 → Core ML logic
├── app.py               → API server
└── requirements.txt

---

📊 Model Insights (Example)

Metric| Score
Accuracy| 0.85
Precision| 0.82
Recall| 0.80

«Replace with actual results»

---

⚠️ Challenges & Learnings

- Data imbalance (few churn users)
- Feature selection importance
- Trade-off between precision & recall

In real-world systems, churn prediction often suffers from imbalanced datasets and shifting user behavior, making evaluation tricky.

---

🚀 Future Improvements

- Handle imbalance (SMOTE / class weights)
- Add explainability (SHAP)
- Real-time prediction system
- Integrate retention strategies
- Deploy on cloud

---

▶️ Run Locally

git clone https://github.com/rohanxlabs/churn-ml-system
cd churn-ml-system
pip install -r requirements.txt
python app.py

---

🌐 API

http://localhost:5000

---

🧑‍💻 Author

Rohan
GitHub: https://github.com/rohanxlabs

---

⭐ Why This Project Stands Out

This project demonstrates:

✔ Real-world ML problem solving
✔ Business-focused thinking
✔ End-to-end ML pipeline

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:FF3C38,100:FF8C42&height=120&section=footer"/>
</p>---

<p align="center">
  <b>“Predicting churn is not about models — it's about saving customers.”</b>
</p>

