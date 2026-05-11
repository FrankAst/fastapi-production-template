from dataclasses import dataclass
from pathlib import Path


@dataclass
class ArtifactPersistError(Exception):
    path: Path
