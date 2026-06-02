from typing import Any

RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {
        "description": "Evaluation artifacts missing — model has not been trained yet."
    },
}
