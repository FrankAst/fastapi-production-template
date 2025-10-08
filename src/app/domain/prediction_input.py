from typing import Self

import numpy as np
from pandas import DataFrame
from pydantic import Field, model_validator

from .base import BaseEntity
from .exceptions import FeaturesContainNaNError, FeaturesEmptyError


class PredictionInput(BaseEntity):
    features: DataFrame = Field(description="Features for prediction")

    @model_validator(mode="after")
    def validate_features(self) -> Self:
        if self.features.empty:
            raise FeaturesEmptyError
        # Check for NaN values
        if np.isnan(self.features).any():
            raise FeaturesContainNaNError

        return self
