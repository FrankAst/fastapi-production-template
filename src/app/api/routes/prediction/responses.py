from typing import Any

from app.domain import ErrorResponse

SINGLE_RESPONSES: dict[int | str, dict[str, Any]] = {
    409: {
        "model": ErrorResponse,
        "description": "Model artifacts or training schema not available.",
    },
    422: {
        "model": ErrorResponse,
        "description": (
            "Request body, Pandera schema, or clinical consistency validation failed."
        ),
    },
    500: {
        "model": ErrorResponse,
        "description": "Internal pipeline configuration error.",
    },
}

BATCH_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {
        "model": ErrorResponse,
        "description": "CSV could not be read or parsed.",
    },
    409: {
        "model": ErrorResponse,
        "description": "Model artifacts or training schema not available.",
    },
    413: {
        "model": ErrorResponse,
        "description": "Upload exceeds the 50 MB cap.",
    },
    422: {
        "model": ErrorResponse,
        "description": (
            "Request, CSV content, Pandera schema, or clinical consistency "
            "validation failed."
        ),
    },
    500: {
        "model": ErrorResponse,
        "description": "Internal pipeline configuration error.",
    },
}
