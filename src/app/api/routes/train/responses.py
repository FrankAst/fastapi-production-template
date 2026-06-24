from typing import Any

from app.domain import ErrorResponse

RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {
        "model": ErrorResponse,
        "description": "CSV could not be read or parsed.",
    },
    409: {
        "model": ErrorResponse,
        "description": "Training schema YAML is missing.",
    },
    413: {
        "model": ErrorResponse,
        "description": "Upload exceeds the 50 MB cap.",
    },
    422: {
        "model": ErrorResponse,
        "description": "Request, CSV content, or Pandera schema validation failed.",
    },
    500: {
        "model": ErrorResponse,
        "description": (
            "Failed to persist a training artifact, or pipeline configuration error."
        ),
    },
}
