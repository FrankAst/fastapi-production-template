from collections.abc import Sequence
from io import StringIO
from typing import Self

import pandas as pd
from fastapi import HTTPException, UploadFile

from app.api.schema import BaseSchema


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
            df: pd.DataFrame = pd.read_csv(StringIO(csv_string))

            # Minimal structure required validation
            if df.shape[1] < 2:
                raise HTTPException(
                    status_code=400, detail="CSV must have at least two columns."
                )

            # Split features and target
            X = df.iloc[:, :-1].to_numpy().tolist()
            y = df.iloc[:, -1].to_numpy().tolist()

            return X, y

        except pd.errors.EmptyDataError as err:
            raise HTTPException(status_code=400, detail="CSV file is empty.") from err
        except pd.errors.ParserError as err:
            raise HTTPException(
                status_code=400, detail="Error parsing CSV file."
            ) from err
        except UnicodeDecodeError as err:
            raise HTTPException(
                status_code=400, detail=f"Error processing CSV: {err!s}"
            ) from err


class TrainResponse(BaseSchema):
    message: str = "Model trained successfully"
