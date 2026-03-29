"""Pandera schema validation for dataset inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandera as pa
from pandera import DataFrameSchema

from app.settings import Settings

from .constants import TARGET_COLUMN
from .exceptions import NoTrainingSchemaError

if TYPE_CHECKING:
    from pathlib import Path

    from pandas import DataFrame


class SchemaValidator:
    """Handles dataset schema validation using Pandera."""

    TRAINING_INPUT_SCHEMA_FILENAME = "training_input_schema.yaml"

    @classmethod
    def get_training_input_schema_path(cls) -> Path:
        """
        Get the path to the static training input schema file.

        Returns:
            Path: Path to the training input schema YAML file.
        """
        return Settings.MODEL_DIRECTORY / cls.TRAINING_INPUT_SCHEMA_FILENAME

    @classmethod
    def validate_training_input(cls, df: DataFrame) -> DataFrame:
        """
        Validate raw training CSV against the static training input schema.

        The static schema covers 12 raw features + the target column.
        strict=False means extra columns are silently ignored, allowing users
        to upload the full dataset without stripping columns first.

        Args:
            df: Raw DataFrame from the training CSV upload.

        Returns:
            DataFrame: Validated and coerced DataFrame.

        Raises:
            NoTrainingSchemaError: If the static schema YAML is missing.
        """
        schema_path = cls.get_training_input_schema_path()
        if not schema_path.exists():
            raise NoTrainingSchemaError
        schema = pa.DataFrameSchema.from_yaml(schema_path)  # pyright: ignore[reportUnknownMemberType]
        return schema.validate(df, lazy=True)

    @classmethod
    def validate_dataframe(cls, df: DataFrame) -> DataFrame:
        """
        Validate prediction input against the static training input schema.

        Loads the same schema used for training CSV validation, strips the target
        column (not present at inference time), and enforces strict=True so that
        any column not in the schema is rejected. This guarantees the prediction
        endpoint receives exactly the raw features the pipeline expects, regardless
        of what extra columns may have been present in the training CSV.

        Args:
            df: Features-only DataFrame from the prediction CSV upload.

        Returns:
            DataFrame: Validated and coerced DataFrame.

        Raises:
            NoTrainingSchemaError: If the static schema YAML is missing.
        """
        schema_path = cls.get_training_input_schema_path()
        if not schema_path.exists():
            raise NoTrainingSchemaError
        schema: DataFrameSchema = pa.DataFrameSchema.from_yaml(schema_path)  # pyright: ignore[reportUnknownMemberType]
        schema = schema.remove_columns([TARGET_COLUMN])
        schema.strict = True
        return schema.validate(df, lazy=True)
