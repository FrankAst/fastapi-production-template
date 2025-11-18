"""Tests for schema validation functionality."""

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from app.domain.exceptions import NoTrainingSchemaError
from app.domain.schema_validator import SchemaValidator


def test_validate_dataframe_no_schema_file(
    valid_prediction_dataframe: pd.DataFrame,
) -> None:
    """
    Test that validate_dataframe raises NoTrainingSchemaError when no schema exists.

    This simulates a user trying to make predictions before training the model.
    The schema file should not exist, and validation should fail appropriately.

    Note: The mock_schema_directory fixture (autouse=True) ensures we're using
    a temporary directory and not touching real schema files.
    """
    # Ensure no schema file exists
    schema_path = SchemaValidator.get_schema_path()
    if schema_path.exists():
        schema_path.unlink()

    # Attempt to validate should raise NoTrainingSchemaError
    with pytest.raises(NoTrainingSchemaError):
        SchemaValidator.validate_dataframe(valid_prediction_dataframe)


def test_validate_dataframe_extra_columns(
    valid_training_dataframe: pd.DataFrame,
    prediction_with_extra_columns: pd.DataFrame,
) -> None:
    """
    Test that validate_dataframe rejects data with extra columns.

    This tests the strict=True enforcement. Prediction data should only
    contain columns that were present in the training data.
    The extra 'experience' column should cause validation to fail.
    """
    # First, create and save a schema from training data
    SchemaValidator.infer_and_save_schema(valid_training_dataframe)

    # Attempt to validate data with extra column should raise SchemaErrors
    with pytest.raises((SchemaError, SchemaErrors)):
        SchemaValidator.validate_dataframe(prediction_with_extra_columns)


def test_validate_dataframe_missing_columns(
    valid_training_dataframe: pd.DataFrame,
    prediction_missing_columns: pd.DataFrame,
) -> None:
    """
    Test that validate_dataframe rejects data with missing required columns.

    Prediction data must contain all columns that were in the training schema.
    Missing the 'education_years' column should cause validation to fail.
    """
    # First, create and save a schema from training data
    SchemaValidator.infer_and_save_schema(valid_training_dataframe)

    # Attempt to validate data with missing column should raise SchemaErrors
    with pytest.raises((SchemaError, SchemaErrors)):
        SchemaValidator.validate_dataframe(prediction_missing_columns)


def test_validate_dataframe_type_coercion_failure(
    valid_training_dataframe: pd.DataFrame,
    prediction_with_wrong_types: pd.DataFrame,
) -> None:
    """
    Test that validate_dataframe rejects data with incompatible types.

    When data types cannot be coerced (e.g., text to int), validation should fail.
    The 'age' column has non-numeric text that cannot convert to int.
    """
    # First, create and save a schema from training data
    SchemaValidator.infer_and_save_schema(valid_training_dataframe)

    # Attempt to validate data with incompatible types should raise SchemaErrors
    with pytest.raises((SchemaError, SchemaErrors)):
        SchemaValidator.validate_dataframe(prediction_with_wrong_types)


def test_validate_dataframe_column_order_independence(
    valid_training_dataframe: pd.DataFrame,
    prediction_with_reordered_columns: pd.DataFrame,
) -> None:
    """
    Test that validate_dataframe accepts data with columns in different order.

    Pandera validates by column name, not position, so columns in a different
    order than the training schema should still validate successfully.
    This is important for real-world scenarios where column order may vary.
    """
    # First, create and save a schema from training data
    SchemaValidator.infer_and_save_schema(valid_training_dataframe)

    # Validate data with reordered columns - should succeed
    result = SchemaValidator.validate_dataframe(prediction_with_reordered_columns)

    # Verify we got a DataFrame back (validation passed)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(prediction_with_reordered_columns)
    # Verify all expected columns are present
    assert set(result.columns) == set(prediction_with_reordered_columns.columns)
