from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def read_yaml(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found at: {path}")

    with path.open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file) or {}

    if not isinstance(content, dict):
        raise ValueError(f"Expected YAML mapping at: {path}")

    return content


def write_json(file_path: str | Path, content: Any) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(content, file, indent=2)
