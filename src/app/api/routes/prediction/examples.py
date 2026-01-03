from fastapi.openapi.models import Example

from .schemas import SinglePredictionRequest

EXAMPLES: dict[str, Example] = {
    "normal": {
        "summary": "A normal example",
        "description": "A **normal** item works correctly.",
        "value": SinglePredictionRequest.create_example().model_dump(by_alias=True),
    },
    "invalid": {
        "summary": "An invalid example",
        "description": "A **invalid** item does not work.",
        "value": SinglePredictionRequest.create_example().model_dump(by_alias=True),
    },
}
