import math
from collections.abc import Sequence
from typing import Self, cast

from fastapi import UploadFile
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
    async def from_upload(cls, file: UploadFile) -> Self:
        """
        Create BatchPredictionRequest from uploaded file.

        Returns:
            BatchPredictionRequest: An instance created from the uploaded file.
        """
        return cls(file=file)

    async def to_feature_matrix(self) -> Sequence[Sequence[float]]:
        """
        Convert uploaded CSV to feature Matrix (no target column).

        Returns:
            Sequence[Sequence[float]]: Feature data.
        """
        df = await process_csv_file(self.file)
        # Convert columns to matrix
        return cast("Sequence[Sequence[float]]", df.to_numpy().tolist())


class SinglePredictionResponse(BaseSchema):
    prediction: float = Field(description="Single prediction result")


class BatchPredictionResponse(BaseSchema):
    predictions: Sequence[float] = Field(description="Array of prediction results")
    count: int = Field(description="Number of predictions made")
