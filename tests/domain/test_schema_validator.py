"""Tests for schema validation functionality."""

import math
from collections.abc import Callable

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from app.domain.constants import TARGET_COLUMN
from app.domain.exceptions import NoTrainingSchemaError
from app.domain.schema_validator import SchemaValidator


def test_validate_dataframe_no_schema_file(
    nhanes_training_dataframe: pd.DataFrame,
) -> None:
    SchemaValidator.get_schema_path().unlink(missing_ok=True)
    with pytest.raises(NoTrainingSchemaError):
        SchemaValidator.validate_dataframe(nhanes_training_dataframe)


def test_validate_dataframe_rejects_extra_columns(
    nhanes_training_dataframe: pd.DataFrame,
) -> None:
    SchemaValidator.infer_and_save_schema(
        nhanes_training_dataframe.drop(columns=[TARGET_COLUMN])
    )
    df = nhanes_training_dataframe.drop(columns=["has_diabetes_or_prediabetes"]).assign(
        extra_col=99.0
    )
    with pytest.raises((SchemaError, SchemaErrors)):
        SchemaValidator.validate_dataframe(df)


def test_validate_dataframe_rejects_missing_columns(
    nhanes_training_dataframe: pd.DataFrame,
) -> None:
    SchemaValidator.infer_and_save_schema(
        nhanes_training_dataframe.drop(columns=[TARGET_COLUMN])
    )
    df = nhanes_training_dataframe.drop(
        columns=["has_diabetes_or_prediabetes", "RIDAGEYR"]
    )
    with pytest.raises((SchemaError, SchemaErrors)):
        SchemaValidator.validate_dataframe(df)


def test_validate_dataframe_rejects_incompatible_types(
    nhanes_training_dataframe: pd.DataFrame,
) -> None:
    SchemaValidator.infer_and_save_schema(
        nhanes_training_dataframe.drop(columns=[TARGET_COLUMN])
    )
    df = nhanes_training_dataframe.drop(columns=["has_diabetes_or_prediabetes"]).assign(
        RIDAGEYR=["not-a-number", "invalid", "bad"]
    )
    with pytest.raises((SchemaError, SchemaErrors)):
        SchemaValidator.validate_dataframe(df)


def test_validate_dataframe_accepts_reordered_columns(
    nhanes_training_dataframe: pd.DataFrame,
) -> None:
    SchemaValidator.infer_and_save_schema(
        nhanes_training_dataframe.drop(columns=[TARGET_COLUMN])
    )
    features = nhanes_training_dataframe.drop(columns=["has_diabetes_or_prediabetes"])
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
