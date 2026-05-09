from fastapi import UploadFile
from pandas import DataFrame
from pydantic import Field

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
    message: str = Field(
        default="Model trained successfully", description="Status message"
    )
    n_samples: int = Field(description="Total rows used for training")
    n_train: int = Field(description="Rows in the training split")
    n_test: int = Field(description="Rows in the held-out test split")
    n_bootstrap: int = Field(
        description="Number of bootstrap resamples in the prediction ensemble"
    )
    threshold: float = Field(description="Decision threshold applied at inference")
