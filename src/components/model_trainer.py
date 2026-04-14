from sklearn.linear_model import LogisticRegression
import joblib
import os

class ModelTrainer:
    def __init__(self):
       self.model_path = "artifacts/model.pkl"
    def train(self,X,y):
      model = LogisticRegression(max_iter=1000)
      model.fit(X,y)
      os.makedirs("artifacts", exist_ok=True)
      model_path = "artifacts/model.pkl"
      joblib.dump(model, model_path)
      return model
    print("model save successfully")
