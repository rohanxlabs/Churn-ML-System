from fastapi import FastAPI
import pandas as pd 
from src.pipeline.predict_pipeline import PredictPipeline
from pydantic import BaseModel
app = FastAPI()
pipeline = PredictPipeline()

class ChurnInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity:str
    OnlineBackup:str
    DeviceProtection:str
    TechSupport:str
    StreamingTV:str
    StreamingMovies:str
    Contract:str
    PaperlessBilling:str
    PaymentMethod:str
    MonthlyCharges:float
    TotalCharges:str

@app.get("/")
def home():
    return {"message": "Churn API is running. Use POST /predict"}

@app.post("/predict")
def predict_churn(data:ChurnInput):
    df = pd.DataFrame([data.model_dump()])
    preds = pipeline.predict(df)
    return{"Churn":"Yes" if preds[0] == 1 else "No"}

