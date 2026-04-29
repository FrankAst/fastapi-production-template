"""Tests for schema validation functionality."""

import math
from collections.abc import Callable

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from app.domain.constants import (
    DIASTOLIC_BP_COLUMN,
    HEIGHT_COLUMN,
    SYSTOLIC_BP_COLUMN,
    TARGET_COLUMN,
    WAIST_COLUMN,
)
from app.domain.exceptions import ClinicalConsistencyError, NoTrainingSchemaError
from app.domain.schema_validator import SchemaValidator


def test_validate_dataframe_no_schema_file(
    nhanes_training_dataframe: pd.DataFrame,
) -> None:
    SchemaValidator.get_training_input_schema_path().unlink(missing_ok=True)
    with pytest.raises(NoTrainingSchemaError):
        SchemaValidator.validate_dataframe(nhanes_training_dataframe)


def test_validate_dataframe_rejects_extra_columns(
    nhanes_training_dataframe: pd.DataFrame,
) -> None:
    df = nhanes_training_dataframe.drop(columns=[TARGET_COLUMN]).assign(extra_col=99.0)
    with pytest.raises((SchemaError, SchemaErrors)):
        SchemaValidator.validate_dataframe(df)


def test_validate_dataframe_rejects_missing_columns(
    nhanes_training_dataframe: pd.DataFrame,
) -> None:
    df = nhanes_training_dataframe.drop(columns=[TARGET_COLUMN, "RIDAGEYR"])
    with pytest.raises((SchemaError, SchemaErrors)):
        SchemaValidator.validate_dataframe(df)


def test_validate_dataframe_rejects_incompatible_types(
    nhanes_training_dataframe: pd.DataFrame,
) -> None:
    df = nhanes_training_dataframe.drop(columns=[TARGET_COLUMN]).assign(
        RIDAGEYR=["not-a-number", "invalid", "bad"]
    )
    with pytest.raises((SchemaError, SchemaErrors)):
        SchemaValidator.validate_dataframe(df)


def test_validate_dataframe_accepts_reordered_columns(
    nhanes_training_dataframe: pd.DataFrame,
) -> None:
    features = nhanes_training_dataframe.drop(columns=[TARGET_COLUMN])
    df = features[list(reversed(features.columns.tolist()))]
    result = SchemaValidator.validate_dataframe(df)
    assert isinstance(result, pd.DataFrame)


def test_validate_training_input_success(
    nhanes_training_dataframe: pd.DataFrame,
) -> None:
    result = SchemaValidator.validate_training_input(nhanes_training_dataframe)
    assert isinstance(result, pd.DataFrame)


def test_validate_training_input_extra_columns_allowed(
    nhanes_training_dataframe: pd.DataFrame,
) -> None:
    df = nhanes_training_dataframe.assign(extra_col=99.0)
    result = SchemaValidator.validate_training_input(df)
    assert isinstance(result, pd.DataFrame)


def test_validate_training_input_nullable_column_accepts_nan(
    nhanes_training_dataframe: pd.DataFrame,
) -> None:
    df = nhanes_training_dataframe.assign(BMXWAIST=math.nan)
    result = SchemaValidator.validate_training_input(df)
    assert isinstance(result, pd.DataFrame)


@pytest.mark.parametrize(
    "invalid_df",
    [
        pytest.param(
            lambda df: df.drop(columns=["RIDAGEYR"]),
            id="missing_required_column",
        ),
        pytest.param(
            lambda df: df.assign(RIDAGEYR=5.0),
            id="age_below_minimum",
        ),
        pytest.param(
            lambda df: df.assign(has_diabetes_or_prediabetes=math.nan),
            id="nan_in_non_nullable_target",
        ),
    ],
)
def test_validate_training_input_rejects_invalid_data(
    nhanes_training_dataframe: pd.DataFrame,
    invalid_df: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    with pytest.raises((SchemaError, SchemaErrors)):
        SchemaValidator.validate_training_input(invalid_df(nhanes_training_dataframe))


def test_validate_training_input_no_schema_file(
    nhanes_training_dataframe: pd.DataFrame,
) -> None:
    SchemaValidator.get_training_input_schema_path().unlink(missing_ok=True)
    with pytest.raises(NoTrainingSchemaError):
        SchemaValidator.validate_training_input(nhanes_training_dataframe)


def test_validate_dataframe_passes_with_consistent_clinical_values(
    nhanes_training_dataframe: pd.DataFrame,
) -> None:
    df = nhanes_training_dataframe.drop(columns=[TARGET_COLUMN])
    result = SchemaValidator.validate_dataframe(df)
    assert isinstance(result, pd.DataFrame)


@pytest.mark.parametrize(
    ("systolic", "diastolic"),
    [
        pytest.param(75.0, 80.0, id="systolic_below_diastolic"),
        pytest.param(80.0, 80.0, id="systolic_equal_to_diastolic"),
    ],
)
def test_validate_dataframe_rejects_invalid_blood_pressure(
    nhanes_training_dataframe: pd.DataFrame,
    systolic: float,
    diastolic: float,
) -> None:
    df = nhanes_training_dataframe.drop(columns=[TARGET_COLUMN]).assign(
        systolic_bp=systolic,
        diastolic_bp=diastolic,
    )
    with pytest.raises(ClinicalConsistencyError):
        SchemaValidator.validate_dataframe(df)


@pytest.mark.parametrize(
    ("waist", "height"),
    [
        pytest.param(180.0, 170.0, id="waist_above_height"),
        pytest.param(170.0, 170.0, id="waist_equal_to_height"),
    ],
)
def test_validate_dataframe_rejects_invalid_body_measurements(
    nhanes_training_dataframe: pd.DataFrame,
    waist: float,
    height: float,
) -> None:
    df = nhanes_training_dataframe.drop(columns=[TARGET_COLUMN]).assign(
        BMXWAIST=waist,
        BMXHT=height,
    )
    with pytest.raises(ClinicalConsistencyError):
        SchemaValidator.validate_dataframe(df)


@pytest.mark.parametrize(
    "nan_assignment",
    [
        pytest.param({"systolic_bp": math.nan}, id="systolic_nan"),
        pytest.param({"diastolic_bp": math.nan}, id="diastolic_nan"),
        pytest.param({"BMXWAIST": math.nan}, id="waist_nan"),
        pytest.param({"BMXHT": math.nan}, id="height_nan"),
    ],
)
def test_validate_dataframe_passes_when_compared_value_is_nan(
    nhanes_training_dataframe: pd.DataFrame,
    nan_assignment: dict[str, float],
) -> None:
    df = nhanes_training_dataframe.drop(columns=[TARGET_COLUMN]).assign(
        **nan_assignment
    )
    result = SchemaValidator.validate_dataframe(df)
    assert isinstance(result, pd.DataFrame)


def test_clinical_failure_cases_records_correct_indices(
    nhanes_training_dataframe: pd.DataFrame,
) -> None:
    df = nhanes_training_dataframe.drop(columns=[TARGET_COLUMN]).assign(
        systolic_bp=[120.0, 70.0, 110.0],
        diastolic_bp=[80.0, 90.0, 70.0],
        BMXWAIST=[95.0, 110.5, 200.0],
        BMXHT=[170.0, 165.0, 180.0],
    )

    with pytest.raises(ClinicalConsistencyError) as captured:
        SchemaValidator.validate_dataframe(df)

    assert captured.value.failure_cases == [
        {
            "column": SYSTOLIC_BP_COLUMN,
            "check": f"greater_than({DIASTOLIC_BP_COLUMN})",
            "index": 1,
        },
        {
            "column": WAIST_COLUMN,
            "check": f"less_than({HEIGHT_COLUMN})",
            "index": 2,
        },
    ]
