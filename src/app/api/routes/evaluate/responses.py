from typing import Any

from app.domain import ErrorResponse

RESPONSES: dict[int | str, dict[str, Any]] = {
    409: {
        "model": ErrorResponse,
        "description": "Evaluation artifacts missing — train the model first.",
    },
}
