from __future__ import annotations

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from src.utils.config import Settings, load_settings
from src.utils.logging import get_logger


logger = get_logger(__name__)


class DataIngestion:
    def __init__(self, settings: Settings | None = None, config_path: str | None = None):
        self.settings = settings or load_settings(config_path)
    
    def ingest(self) -> pd.DataFrame:
        logger.info("Loading dataset from %s", self.settings.raw_data_path)
        try:
            return pd.read_csv(self.settings.raw_data_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Dataset not found at {self.settings.raw_data_path}. "
                "Update config/config.yaml or CHURN_CONFIG_PATH to a valid dataset path."
            ) from exc
        except EmptyDataError as exc:
            raise ValueError(f"Dataset at {self.settings.raw_data_path} is empty.") from exc
        except ParserError as exc:
            raise ValueError(
                f"Dataset at {self.settings.raw_data_path} could not be parsed as CSV."
            ) from exc
