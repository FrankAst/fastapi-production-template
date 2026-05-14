from typing import Any

SINGLE_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"description": "Model artifacts missing — train the model first."},
    422: {"description": "Request body or clinical-consistency validation failed."},
}

BATCH_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {
        "description": "CSV validation failed, or model artifacts missing — "
        "train the model first."
    },
    413: {"description": "Upload exceeds the 50 MB cap."},
    422: {"description": "Pandera schema or clinical-consistency validation failed."},
}
