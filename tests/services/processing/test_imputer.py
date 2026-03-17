import pandas as pd
import pytest

from app.services.processing.imputer.config import (
    MEDIAN_FILL_COLS,
    PASSTHROUGH_COLS,
    ZERO_FILL_COLS,
)
from app.services.processing.imputer.transformer import Imputer


def test_no_nans_after_transform(imputer_dataframe_with_nans: pd.DataFrame) -> None:
    result = (
        Imputer()
        .fit(imputer_dataframe_with_nans)
        .transform(imputer_dataframe_with_nans)
    )

    assert result.isna().sum().sum() == 0


def test_zero_fill_columns(imputer_dataframe_with_nans: pd.DataFrame) -> None:
    # vigorous_minutes_per_week[1] and drinking_frequency[2] are NaN in the fixture
    result = (
        Imputer()
        .fit(imputer_dataframe_with_nans)
        .transform(imputer_dataframe_with_nans)
    )

    assert result["vigorous_minutes_per_week"].iloc[1] == pytest.approx(0.0)  # pyright: ignore[reportUnknownMemberType]
    assert result["drinking_frequency"].iloc[2] == pytest.approx(0.0)  # pyright: ignore[reportUnknownMemberType]


def test_median_fill_columns(imputer_dataframe_with_nans: pd.DataFrame) -> None:
    # waist_to_height_ratio fixture values: [0.5, NaN, 0.7] → median = 0.6
    result = (
        Imputer()
        .fit(imputer_dataframe_with_nans)
        .transform(imputer_dataframe_with_nans)
    )

    assert result["waist_to_height_ratio"].iloc[1] == pytest.approx(0.6)  # pyright: ignore[reportUnknownMemberType]


def test_passthrough_columns_unchanged(
    imputer_dataframe_with_nans: pd.DataFrame,
) -> None:
    result = (
        Imputer()
        .fit(imputer_dataframe_with_nans)
        .transform(imputer_dataframe_with_nans)
    )

    for col in PASSTHROUGH_COLS:
        assert result[col].tolist() == imputer_dataframe_with_nans[col].tolist()


def test_fit_transform_separation(
    imputer_fit_dataframe: pd.DataFrame,
    imputer_transform_dataframe: pd.DataFrame,
) -> None:
    # fit data: diastolic_bp = [60, 80, 100] → median = 80
    # transform data: diastolic_bp = [NaN, 200, 200]
    # NaN should be filled with 80 (fit median), not 200 (transform median)
    imputer = Imputer()
    imputer.fit(imputer_fit_dataframe)
    result = imputer.transform(imputer_transform_dataframe)

    assert result["diastolic_bp"].iloc[0] == pytest.approx(80.0)  # pyright: ignore[reportUnknownMemberType]


def test_no_nan_input_unchanged(imputer_dataframe_no_nans: pd.DataFrame) -> None:
    result = (
        Imputer().fit(imputer_dataframe_no_nans).transform(imputer_dataframe_no_nans)
    )

    for col in ZERO_FILL_COLS + MEDIAN_FILL_COLS + PASSTHROUGH_COLS:
        assert result[col].tolist() == pytest.approx(  # pyright: ignore[reportUnknownMemberType]
            imputer_dataframe_no_nans[col].tolist()
        )


def test_column_names_preserved(imputer_dataframe_with_nans: pd.DataFrame) -> None:
    result = (
        Imputer()
        .fit(imputer_dataframe_with_nans)
        .transform(imputer_dataframe_with_nans)
    )

    expected_cols = ZERO_FILL_COLS + MEDIAN_FILL_COLS + PASSTHROUGH_COLS
    assert list(result.columns) == expected_cols
