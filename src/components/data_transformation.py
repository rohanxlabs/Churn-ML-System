import joblib 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataTransformation:
    def transform(self,df,target_col):
        df = df.drop(columns=["customerID"])

        X = df.drop(columns=[target_col])
        y = df[target_col].map({"Yes":1,"No":0})

        joblib.dump(X.columns.tolist(), "artifacts/feature_names.pkl")

        num_cols = X.select_dtypes(include=["int64","float64"]).columns
        cat_cols = X.select_dtypes(include=["object"]).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(),num_cols),
                ("cat",OneHotEncoder(handle_unknown="ignore"),cat_cols)
            ]
        )

        X_transformed = preprocessor.fit_transform(X)

        joblib.dump(preprocessor, "artifacts/preprocessor.pkl")

        return X_transformed ,y