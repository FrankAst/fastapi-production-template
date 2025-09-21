import math
from collections.abc import Sequence
from io import StringIO
from typing import Self, cast

import pandas as pd
from fastapi import HTTPException, UploadFile
from pandas import DataFrame
from pydantic import ConfigDict, Field, model_validator

from app.api.schema import BaseSchema


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
        Raises:
            HTTPException: If file is not CSV or processing fails.
        """
        if not self.file.filename or not self.file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported.")

        try:
            # Read CSV content
            contents = await self.file.read()
            csv_string = contents.decode("utf-8")

            # Parse CSV using pandas
            df: DataFrame = pd.read_csv(StringIO(csv_string))  # type: ignore[misc]

            # Convert columns to matrix
            return cast("Sequence[Sequence[float]]", df.to_numpy().tolist())

        except Exception as err:
            raise HTTPException(
                status_code=400, detail=f"Error processing CSV: {err!s}"
            ) from err


class SinglePredictionResponse(BaseSchema):
    prediction: float = Field(description="Single prediction result")


class BatchPredictionResponse(BaseSchema):
    predictions: Sequence[float] = Field(description="Array of prediction results")
    count: int = Field(description="Number of predictions made")
