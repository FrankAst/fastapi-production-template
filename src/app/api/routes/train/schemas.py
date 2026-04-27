from fastapi import UploadFile
from pandas import DataFrame

from app.api.schema import BaseSchema
from app.domain import SchemaValidator
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
            DataFrame: Validated and coerced training DataFrame.
        """
        df = await process_csv_file(file)
        return SchemaValidator.validate_training_input(df)


class TrainResponse(BaseSchema):
    message: str = "Model trained successfully"
    n_samples: int
    n_train: int
    n_test: int
    n_bootstrap: int
    n_shap_background: int
    threshold: float
