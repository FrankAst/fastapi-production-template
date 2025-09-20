import math
from collections.abc import Sequence
from io import StringIO
from typing import Self

import pandas as pd
from fastapi import HTTPException, UploadFile
from pydantic import Field, model_validator

from app.api.schema import BaseSchema
from app.domain.exceptions import FeaturesContainNaNError, FeaturesEmptyError


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

    class Config:
        arbitrary_types_allowed = True

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
            HTTPException: If the uploaded file is not a CSV, is empty, has less than
            one column, or cannot be parsed.
            FeaturesEmptyError: If the CSV file has less than one column.
            FeaturesContainNaNError: If the CSV file contains NaN values.
        """
        if not self.file.filename or not self.file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported.")

        try:
            # Read CSV content
            contents = await self.file.read()
            csv_string = contents.decode("utf-8")

            # Parse CSV using pandas
            df: pd.DataFrame = pd.read_csv(StringIO(csv_string))
            if df.shape[1] < 1:
                raise FeaturesEmptyError

            # Check for NaN values
            if df.isna().to_numpy().any():
                raise FeaturesContainNaNError

            # Convert columns to matrix
            return df.to_numpy().tolist()

        except pd.errors.EmptyDataError as err:
            raise HTTPException(status_code=400, detail="CSV file is empty") from err
        except pd.errors.ParserError as err:
            raise HTTPException(status_code=400, detail="Invalid CSV format") from err
        except UnicodeDecodeError as err:
            raise HTTPException(
                status_code=400, detail="Uploaded file could not be decoded as UTF-8"
            ) from err
        except Exception as err:
            raise HTTPException(
                status_code=400, detail=f"Error processing CSV: {err!s}"
            ) from err


class SinglePredictionResponse(BaseSchema):
    prediction: float = Field(description="Single prediction result")


class BatchPredictionResponse(BaseSchema):
    predictions: Sequence[float] = Field(description="Array of prediction results")
    count: int = Field(description="Number of predictions made")
