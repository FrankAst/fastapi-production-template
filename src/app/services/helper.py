from pathlib import Path
from typing import Any

import joblib

from app.domain.ml_model import MLModel

from .exceptions import ArtifactPersistError


def load_model(model_path: Path) -> MLModel | None:
    if not model_path.exists():
        return None

    return joblib.load(model_path)  # type: ignore[no-any-return]


def save_model(model: MLModel, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(model, model_path)
    except OSError as e:
        raise ArtifactPersistError(model_path) from e


def save_artifact(artifact: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(artifact, path)
    except OSError as e:
        raise ArtifactPersistError(path) from e
