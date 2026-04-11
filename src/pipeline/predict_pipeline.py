import joblib 
import pandas as pd 

class PredictPipeline:
    def __init__(self):
        self.model= joblib.load("artifacts/model.pkl")
        self.preprocessor = joblib.load("artifacts/preprocessor.pkl")
        self.feature_names = joblib.load("artifacts/feature_names.pkl")
        
    def predict(self,data:pd.DataFrame):

    
        if "customerID" in data.columns:
            data = data.drop(columns=["customerID"])

        missing = [col for col in self.feature_names if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        data = data[self.feature_names]

        X = self.preprocessor.transform(data)
        preds = self.model.predict(X)
        return preds
        