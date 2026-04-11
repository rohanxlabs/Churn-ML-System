import pandas as pd
from src.utils.common import read_yaml

class DataIngestion:
    def __init__(self, config_path):
        self.config = read_yaml(config_path)
    
    def ingest(self):
        path = self.config["data"]["raw_data_path"]
        df = pd.read_csv(path)
        return df