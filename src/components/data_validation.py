class DataValidation:
    def validate(self, df):
      assert "Churn" in df.columns
      assert df.isnull().sum().sum() == 0
      return True