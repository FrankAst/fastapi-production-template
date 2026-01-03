from fastapi import UploadFile
from pandas import DataFrame

from app.api.schema import BaseSchema
from app.utils import process_csv_file


class FileTrainRequest(BaseSchema):
    file: UploadFile

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    async def from_upload(cls, file: UploadFile) -> DataFrame:
        """
        Create FileTrainRequest from uploaded file.

        Returns:
            FileTrainRequest: An instance created from the uploaded file.
        """
        return await process_csv_file(file)


class TrainResponse(BaseSchema):
    message: str = "Model trained successfully"
