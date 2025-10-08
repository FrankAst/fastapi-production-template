import math
from collections.abc import Sequence
from typing import Self

from fastapi import UploadFile
from pandas import DataFrame
from pydantic import ConfigDict, Field, model_validator

from app.api.schema import BaseSchema
from app.utils import process_csv_file


class SinglePredictionRequest(BaseSchema):
    features: Sequence[float] = Field(
        description="Array of features for prediction", min_length=1
    )

    @model_validator(mode="after")
    def validate_features(self) -> Self:
        # Check for NaN values
        if any(math.isnan(feature) for feature in self.features):
            msg = "Features list must not contain NaN values"
            raise ValueError(msg)
        return self


class BatchPredictionRequest(BaseSchema):
    file: UploadFile

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    async def from_upload(cls, file: UploadFile) -> DataFrame:
        """
        Create feature matrix from uploaded file.

        Returns:
            DataFrame: Processed feature data from the uploaded file.
        """
        return await process_csv_file(file)


class SinglePredictionResponse(BaseSchema):
    prediction: float = Field(description="Single prediction result")


class BatchPredictionResponse(BaseSchema):
    predictions: Sequence[float] = Field(description="Array of prediction results")
    count: int = Field(description="Number of predictions made")
