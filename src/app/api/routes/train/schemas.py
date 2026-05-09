from fastapi import UploadFile
from pandas import DataFrame

from app.api.schema import BaseSchema
from app.domain import SchemaValidator
from app.utils import process_csv_file


async def parse_training_upload(file: UploadFile) -> DataFrame:
    """Parse and validate an uploaded CSV against the training schema.

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
    threshold: float
