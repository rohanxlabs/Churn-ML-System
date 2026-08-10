from __future__ import annotations

import pytest

from src.components.data_ingestion import DataIngestion


def test_data_ingestion_reports_missing_dataset(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  raw_data_path: missing.csv",
                "artifacts:",
                "  dir: artifacts",
                "model:",
                "  name: logistic_regression",
                "training:",
                "  test_size: 0.2",
                "  random_state: 42",
                "logging:",
                "  level: INFO",
            ]
        ),
        encoding="utf-8",
    )

    ingestion = DataIngestion(config_path=str(config_path))

    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        ingestion.ingest()
