from pathlib import Path

from app.domain import AppError


class ArtifactPersistError(AppError):
    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__("Failed to persist artifact")
