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


def test_load_training_schema_returns_cached_object() -> None:
    # pylint: disable=protected-access
    schema_path = SchemaValidator.get_training_input_schema_path()
    first = SchemaValidator._load_training_schema(schema_path)  # noqa: SLF001
    second = SchemaValidator._load_training_schema(schema_path)  # noqa: SLF001
    assert first is second


def test_validate_dataframe_does_not_mutate_cached_base_schema(
    nhanes_training_dataframe: pd.DataFrame,
) -> None:
    # pylint: disable=protected-access
    schema_path = SchemaValidator.get_training_input_schema_path()
    base_before = SchemaValidator._load_training_schema(schema_path)  # noqa: SLF001
    columns_before = list(base_before.columns)
    strict_before = base_before.strict

    features = nhanes_training_dataframe.drop(columns=[TARGET_COLUMN])
    SchemaValidator.validate_dataframe(features)
    SchemaValidator.validate_dataframe(features)

    base_after = SchemaValidator._load_training_schema(schema_path)  # noqa: SLF001
    assert base_after is base_before
    assert list(base_after.columns) == columns_before
    assert TARGET_COLUMN in base_after.columns
    assert base_after.strict == strict_before
