from collections.abc import Sequence
from typing import Self

from fastapi import HTTPException, UploadFile

from app.api.schema import BaseSchema
from app.domain import MIN_REQUIRED_COLUMNS
from app.utils import process_csv_file


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
        df = await process_csv_file(self.file)

        # Minimal structure required validation
        if df.shape[1] < MIN_REQUIRED_COLUMNS:
            raise HTTPException(
                status_code=400, detail="CSV must have at least two columns."
            )
        # Split features and target
        X = df.iloc[:, :-1].to_numpy().tolist()
        y = df.iloc[:, -1].to_numpy().tolist()

        return X, y


class TrainResponse(BaseSchema):
    message: str = "Model trained successfully"
