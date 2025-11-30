"""Pandera schema validation for dataset inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandera as pa
from pandera import DataFrameSchema

from app.settings import Settings

from .exceptions import NoTrainingSchemaError

if TYPE_CHECKING:
    from pathlib import Path

    from pandas import DataFrame


class SchemaValidator:
    """Handles dataset schema validation using Pandera."""

    SCHEMA_FILENAME = "training_schema.yaml"

    @classmethod
    def get_schema_path(cls) -> Path:
        """
        Get the path to the saved schema file.

        Returns:
            Path: Path to the training schema YAML file.
        """
        return Settings.MODEL_DIRECTORY / cls.SCHEMA_FILENAME

    @classmethod
    def infer_and_save_schema(cls, df: DataFrame) -> DataFrameSchema:
        """
        Infer Pandera schema from training data and save it.

        This method creates a schema from the feature columns (excluding target),
        configured to be lenient for prediction use:
        - Coerce types when possible
        - Allow missing values (nullable=True for all columns)

        Args:
            df: Training DataFrame with features and target column.
                The last column is assumed to be the target.

        Returns:
            DataFrameSchema: The inferred schema.
        """
        # Get feature columns only (exclude target - last column)
        features_df = df.iloc[:, :-1]

        # Infer base schema from the features
        schema = pa.infer_schema(features_df)

        # Make schema lenient for prediction:
        # 1. Enable type coercion
        # 2. Make all columns nullable (allow missing values)
        for column_schema in schema.columns.values():
            column_schema.nullable = True
            column_schema.coerce = True

        # Configure DataFrame-level settings
        schema.coerce = True  # Enable coercion at DataFrame level
        schema.strict = True  # Forbid columns not in schema

        # Save schema to YAML for reuse
        schema_path = cls.get_schema_path()
        schema.to_yaml(schema_path)  # pyright: ignore[reportUnknownMemberType]

        return schema

    @classmethod
    def load_schema(cls) -> DataFrameSchema | None:
        """
        Load the saved Pandera schema from disk.

        Returns:
            DataFrameSchema if schema file exists, None otherwise.
        """
        schema_path = cls.get_schema_path()
        if not schema_path.exists():
            return None

        return pa.DataFrameSchema.from_yaml(schema_path)  # pyright: ignore[reportUnknownMemberType]

    @classmethod
    def validate_dataframe(cls, df: DataFrame) -> DataFrame:
        """
        Validate a DataFrame against the saved schema.

        This validates prediction input data against the schema created
        from raw training dataset. The validation is lenient:
        - Attempts type coercion
        - Allows missing values
        - Forbids extra columns not in the training schema

        Args:
            df: DataFrame to validate (raw prediction input).

        Returns:
            DataFrame: Validated and coerced DataFrame.

        Raises:
            NoTrainingSchemaError: If no schema file exists (model not trained).
        """
        schema = cls.load_schema()

        if schema is None:
            raise NoTrainingSchemaError

        # Validate and return the coerced DataFrame
        # Pandera will raise SchemaError with detailed information if validation fails
        return schema.validate(df, lazy=True)
