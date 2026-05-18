from typing import Any

RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"description": "CSV validation failed (filename, columns, or target)."},
    413: {"description": "Upload exceeds the 50 MB cap."},
    422: {"description": "Pandera schema validation failed."},
    500: {"description": "Failed to persist a training artifact."},
}
