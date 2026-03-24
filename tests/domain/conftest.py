"""Fixtures for domain layer tests."""

import math
import shutil
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


@pytest.fixture
def nhanes_training_dataframe() -> pd.DataFrame:
    """13-column NHANES training DataFrame (12 features + target).

    Returns:
        DataFrame with 3 rows and the expected NHANES columns.
    """

    return pd.DataFrame({
        "RIDAGEYR": [45.0, 62.0, 28.0],
        "BMXWAIST": [95.0, 110.5, 78.0],
        "BMXHT": [170.0, 165.0, 180.0],
        "told_high_bp": [1.0, 0.0, 0.0],
        "told_high_cholesterol": [0.0, 1.0, math.nan],
        "is_female": [1.0, 0.0, 1.0],
        "drinking_frequency": [2.0, 0.0, 3.0],
        "diastolic_bp": [80.0, 90.0, 70.0],
        "systolic_bp": [120.0, 145.0, 110.0],
        "education_level": [4.0, 3.0, 5.0],
        "phq9_score": [5.0, 12.0, 0.0],
        "vigorous_minutes_per_week": [0.0, 60.0, 150.0],
        "has_diabetes_or_prediabetes": [0.0, 1.0, 0.0],
    })


@pytest.fixture(autouse=True)
def mock_schema_directory(tmp_path: Path) -> Generator[None]:
    """Mock both schema paths to isolated temp directories."""
    schema_path = tmp_path / "ml_binaries" / "training_schema.yaml"
    training_input_schema_path = tmp_path / "ml_binaries" / "training_input_schema.yaml"
    schema_path.parent.mkdir(parents=True, exist_ok=True)

    real_schema = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "app"
        / "ml_binaries"
        / "training_input_schema.yaml"
    )
    if real_schema.exists():
        shutil.copy(real_schema, training_input_schema_path)

    with (
        patch(
            "app.domain.schema_validator.SchemaValidator.get_schema_path",
            return_value=schema_path,
        ),
        patch(
            "app.domain.schema_validator.SchemaValidator.get_training_input_schema_path",
            return_value=training_input_schema_path,
        ),
    ):
        yield
