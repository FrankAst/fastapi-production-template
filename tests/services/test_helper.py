from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.services.exceptions import ArtifactPersistError
from app.services.helper import save_artifact, save_model


def test_save_artifact_creates_parent_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "subdir" / "artifact.joblib"
    assert not target.parent.exists()

    save_artifact({"key": "value"}, target)

    assert target.parent.exists()
    assert target.exists()


@pytest.mark.parametrize(
    ("save_fn", "payload"),
    [
        pytest.param(save_artifact, {"key": "value"}, id="save_artifact"),
        pytest.param(save_model, MagicMock(), id="save_model"),
    ],
)
def test_save_raises_artifact_persist_error_on_os_error(
    save_fn: Callable[..., None],
    payload: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact.joblib"
    monkeypatch.setattr(
        "app.services.helper.joblib.dump",
        MagicMock(side_effect=OSError("write failed")),
    )

    with pytest.raises(ArtifactPersistError) as exc_info:
        save_fn(payload, target)

    assert exc_info.value.path == target
