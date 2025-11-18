"""Fixtures for domain layer tests."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


@pytest.fixture
def valid_training_dataframe() -> pd.DataFrame:
    """
    Create a valid training DataFrame with features and target.

    Structure: 3 feature columns + 1 target column (last).

    Returns:
        DataFrame with training data.
    """
    return pd.DataFrame({
        "age": [25, 30, 35, 40],
        "income": [50000, 60000, 70000, 80000],
        "education_years": [12, 16, 18, 20],
        "target": [0, 1, 1, 0],
    })


@pytest.fixture
def valid_prediction_dataframe() -> pd.DataFrame:
    """
    Create a valid prediction DataFrame matching the training schema.

    Has the same columns as training (minus target).

    Returns:
        DataFrame with prediction data.
    """
    return pd.DataFrame({
        "age": [28, 32],
        "income": [55000, 65000],
        "education_years": [14, 17],
    })


@pytest.fixture
def prediction_with_extra_columns() -> pd.DataFrame:
    """
    Prediction DataFrame with an extra column not in training schema.

    This should fail validation due to strict=True.

    Returns:
        DataFrame with extra column.
    """
    return pd.DataFrame({
        "age": [28, 32],
        "income": [55000, 65000],
        "education_years": [14, 17],
        "experience": [2, 5],  # Extra column not in training
    })


@pytest.fixture
def prediction_missing_columns() -> pd.DataFrame:
    """
    Prediction DataFrame missing a required column from training schema.

    Missing 'education_years' column.

    Returns:
        DataFrame missing required column.
    """
    return pd.DataFrame({
        "age": [28, 32],
        "income": [55000, 65000],
    })


@pytest.fixture
def prediction_with_wrong_types() -> pd.DataFrame:
    """
    Prediction DataFrame with incompatible types that cannot be coerced.

    'age' column has text values that cannot convert to int.

    Returns:
        DataFrame with incompatible types.
    """
    return pd.DataFrame({
        "age": ["not-a-number", "invalid-age"],
        "income": [55000, 65000],
        "education_years": [14, 17],
    })


@pytest.fixture
def prediction_with_reordered_columns() -> pd.DataFrame:
    """
    Prediction DataFrame with columns in different order than training.

    Should still validate successfully as Pandera validates by column name.

    Returns:
        DataFrame with reordered columns.
    """
    return pd.DataFrame({
        "education_years": [14, 17],
        "age": [28, 32],
        "income": [55000, 65000],
    })


@pytest.fixture(autouse=True)
def mock_schema_directory(
    tmp_path: Path,
) -> Generator[None, None, None]:
    """
    Mock the schema directory to use a temporary path during tests.

    This ensures tests don't interfere with the real model schema files.
    The fixture is autouse=True so it applies to all tests automatically.

    We mock SchemaValidator.get_schema_path() to return a temp path.

    Yields:
        None
    """
    # Create schema path in temp directory
    test_schema_path = tmp_path / "ml_binaries" / "training_schema.yaml"
    test_schema_path.parent.mkdir(parents=True, exist_ok=True)

    # Mock get_schema_path to return our temp path
    with patch(
        "app.domain.schema_validator.SchemaValidator.get_schema_path",
        return_value=test_schema_path,
    ):
        yield
