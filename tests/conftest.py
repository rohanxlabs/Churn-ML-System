from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.pipeline.train_pipeline import run_training


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "churn.csv"


@pytest.fixture(scope="session")
def trained_artifact_dir(tmp_path_factory):
    artifact_dir = tmp_path_factory.mktemp("artifacts")
    previous_artifact_dir = os.environ.get("CHURN_ARTIFACT_DIR")

    os.environ["CHURN_ARTIFACT_DIR"] = str(artifact_dir)
    run_training()

    yield artifact_dir

    if previous_artifact_dir is None:
        os.environ.pop("CHURN_ARTIFACT_DIR", None)
    else:
        os.environ["CHURN_ARTIFACT_DIR"] = previous_artifact_dir
