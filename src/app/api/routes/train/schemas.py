from collections.abc import Sequence
from io import StringIO
from typing import Self

import pandas as pd
from fastapi import HTTPException, UploadFile
from pandas import DataFrame

from app.api.schema import BaseSchema
from app.domain import MIN_REQUIRED_COLUMNS


class FileTrainRequest(BaseSchema):
    file: UploadFile

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    async def from_upload(cls, file: UploadFile) -> Self:
        """
        Create FileTrainRequest from uploaded file.

        Returns:
            FileTrainRequest: An instance created from the uploaded file.
        """
        return cls(file=file)

    async def to_training_data(self) -> tuple[Sequence[Sequence[float]], Sequence[int]]:
        """Convert uploaded CSV to training data format - last column is target

        Returns:
            tuple[Sequence[Sequence[float]], Sequence[int]]: Features and target data.

        Raises:
            HTTPException: If the uploaded file is not a CSV, is empty, has less than
            two columns, or cannot be parsed.
        """

        # Validate file type
        if not self.file.filename or not self.file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported.")

        try:
            # Read CSV content
            contents = await self.file.read()
            csv_string = contents.decode("utf-8")
            # Parse CSV using pandas
            df: DataFrame = pd.read_csv(StringIO(csv_string))  # type: ignore[misc]

        except Exception as err:
            raise HTTPException(
                status_code=400, detail=f"Error processing CSV: {err!s}"
            ) from err

        # Minimal structure required validation
        if df.shape[1] < MIN_REQUIRED_COLUMNS:
            raise HTTPException(
                status_code=400, detail="CSV must have at least two columns."
            )
        # Split features and target
        X = df.iloc[:, :-1].to_numpy().tolist()
        y = df.iloc[:, -1].to_numpy().tolist()  # type: ignore[assignment]

        return X, y


class TrainResponse(BaseSchema):
    message: str = "Model trained successfully"
