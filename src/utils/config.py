from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.utils.common import read_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def _resolve_path(path_value: str | Path, base_path: Path = PROJECT_ROOT) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base_path / path).resolve()


@dataclass(frozen=True)
class Settings:
    project_root: Path
    config_path: Path
    raw_data_path: Path
    artifacts_dir: Path
    model_path: Path
    preprocessor_path: Path
    feature_names_path: Path
    metrics_path: Path
    validation_report_path: Path
    model_name: str
    test_size: float
    random_state: int
    class_weight: str | None
    log_level: str


def load_settings(config_path: str | Path | None = None) -> Settings:
    resolved_config_path = _resolve_path(
        os.getenv("CHURN_CONFIG_PATH", str(config_path or DEFAULT_CONFIG_PATH))
    )
    config = read_yaml(resolved_config_path)

    artifact_dir_value = os.getenv(
        "CHURN_ARTIFACT_DIR",
        config.get("artifacts", {}).get("dir", "artifacts"),
    )
    artifacts_dir = _resolve_path(artifact_dir_value)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        project_root=PROJECT_ROOT,
        config_path=resolved_config_path,
        raw_data_path=_resolve_path(config["data"]["raw_data_path"]),
        artifacts_dir=artifacts_dir,
        model_path=artifacts_dir / "model.pkl",
        preprocessor_path=artifacts_dir / "preprocessor.pkl",
        feature_names_path=artifacts_dir / "feature_names.pkl",
        metrics_path=artifacts_dir / "metrics.json",
        validation_report_path=artifacts_dir / "validation_report.json",
        model_name=config.get("model", {}).get("name", "logistic_regression"),
        test_size=float(config.get("training", {}).get("test_size", 0.2)),
        random_state=int(config.get("training", {}).get("random_state", 42)),
        class_weight=config.get("model", {}).get("class_weight"),
        log_level=os.getenv("CHURN_LOG_LEVEL", config.get("logging", {}).get("level", "INFO")),
    )
