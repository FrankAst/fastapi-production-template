import pytest
from pydantic import ValidationError

from app.domain import BatchPredictionOutput


def test_instantiates_with_predictions_and_count() -> None:
    predictions = [0.1, 0.7, 0.4]

    output = BatchPredictionOutput(predictions=predictions, count=len(predictions))

    assert output.predictions == pytest.approx(predictions)  # pyright: ignore[reportUnknownMemberType]
    assert output.count == len(predictions)


def test_rejects_non_positive_count() -> None:
    with pytest.raises(ValidationError):
        BatchPredictionOutput(predictions=[], count=0)
