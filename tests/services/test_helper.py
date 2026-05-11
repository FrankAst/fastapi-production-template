from pathlib import Path

from app.services.helper import save_artifact


def test_save_artifact_creates_parent_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "subdir" / "artifact.joblib"
    assert not target.parent.exists()

    save_artifact({"key": "value"}, target)

    assert target.parent.exists()
    assert target.exists()
