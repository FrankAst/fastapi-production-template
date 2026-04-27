from collections.abc import Sequence

from pydantic import Field

from .base import BaseEntity


class BatchPredictionOutput(BaseEntity):
    predictions: Sequence[float] = Field(description="Array of prediction results")
    count: int = Field(gt=0, description="Number of predictions made")
