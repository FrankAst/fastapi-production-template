"""Fixtures for preprocessing tests."""

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def valid_training_data() -> dict[str, list[Any]]:
    """
    Raw training data as dictionary.

    Returns:
        Dictionary with feature columns and target.
    """
    return {
        "RIDAGEYR": [25, 30, 35, 40],
        "income": [50000, 60000, 70000, 80000],
        "education_years": [12, 16, 18, 20],
        "target": [0, 1, 1, 0],
    }


@pytest.fixture
def valid_training_dataframe(valid_training_data: dict[str, list[Any]]) -> pd.DataFrame:
    """
    Create a valid training DataFrame with features and target.

    Args:
        valid_training_data: Raw training data dictionary.

    Returns:
        DataFrame with training data.
    """
    return pd.DataFrame(valid_training_data)


@pytest.fixture
def dataframe_with_all_age_categories(
    valid_training_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    DataFrame with RIDAGEYR values covering all age categories.

    Args:
        valid_training_dataframe: Base DataFrame with training data.

    Returns:
        DataFrame with ages [18, 45, 65, 80] representing each category.
    """
    df = valid_training_dataframe.copy()
    df["RIDAGEYR"] = [18, 45, 65, 80]
    return df


@pytest.fixture
def age_categories() -> list[str]:
    """
    One-hot encoded column names produced by AgeBinner.

    Returns:
        List of age category column names.
    """
    return ["young_adult", "middle_age", "senior", "elderly", "age_unknown"]


def _age_test_cases() -> list[Any]:
    """
    Age values with expected category mappings.

    Age semantics:
    - 18-44  -> young_adult
    - 45-64  -> middle_age
    - 65-79  -> senior
    - 80     -> elderly (top-coded)
    - NaN    -> age_unknown

    Returns:
        List of pytest.param objects with (age, expected_category) tuples.
    """

    cases: list[tuple[float, str, str]] = [
        (18, "young_adult", "young_adult_lower"),
        (30, "young_adult", "young_adult_mid"),
        (44, "young_adult", "young_adult_upper"),
        (45, "middle_age", "middle_age_lower"),
        (55, "middle_age", "middle_age_mid"),
        (64, "middle_age", "middle_age_upper"),
        (65, "senior", "senior_lower"),
        (72, "senior", "senior_mid"),
        (79, "senior", "senior_upper"),
        (80, "elderly", "elderly_top_coded"),
        (np.nan, "age_unknown", "unknown_nan"),
    ]

    return [
        pytest.param((age, category), id=case_id) for age, category, case_id in cases
    ]


@pytest.fixture(params=_age_test_cases())
def age_with_expected_category(
    request: pytest.FixtureRequest,
) -> tuple[float | None, str]:
    """
    Parametrized fixture yielding (age, expected_category) tuples.

    Args:
        request: Pytest fixture request object.

    Returns:
        Tuple of (age_value, expected_category_name).
    """
    return cast("tuple[float | None, str]", request.param)


# ---------------------------------------------------------------------------
# WaistToHeightRatio fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def waist_height_dataframe() -> pd.DataFrame:
    """
    DataFrame with BMXWAIST, BMXHT, and an unrelated column.

    Ratios are easy to verify by hand:
    - Row 0: 90 / 180 = 0.5
    - Row 1: 96 / 160 = 0.6

    Returns:
        DataFrame with two rows of anthropometric data plus an unrelated column.
    """
    return pd.DataFrame({
        "BMXWAIST": [90.0, 96.0],
        "BMXHT": [180.0, 160.0],
        "other_col": [1, 2],
    })


def _waist_height_edge_cases() -> list[Any]:
    """
    Edge-case (waist, height, expected_ratio) tuples for WaistToHeightRatio.

    Returns:
        List of pytest.param objects.
    """
    cases: list[tuple[float | None, float | None, float | None, str]] = [
        (90.0, 180.0, 0.5, "normal_values"),
        (100.0, 0.0, None, "zero_height"),
        (None, 170.0, None, "nan_waist"),
        (95.0, None, None, "nan_height"),
        (None, None, None, "both_nan"),
    ]
    return [
        pytest.param((waist, height, expected), id=case_id)
        for waist, height, expected, case_id in cases
    ]


@pytest.fixture(params=_waist_height_edge_cases())
def edge_case_waist_height(
    request: pytest.FixtureRequest,
) -> tuple[float | None, float | None, float | None]:
    """
    Parametrized fixture yielding (waist, height, expected_ratio) tuples.

    expected_ratio is None where a NaN result is expected.

    Args:
        request: Pytest fixture request object.

    Returns:
        Tuple of (waist_value, height_value, expected_ratio).
    """
    return cast("tuple[float | None, float | None, float | None]", request.param)


# ---------------------------------------------------------------------------
# CholesterolMissingness fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cholesterol_dataframe() -> pd.DataFrame:
    """
    DataFrame with told_high_cholesterol containing mixed values and
    an unrelated column.

    NaN at index 2 → expected missingness flag: [0, 0, 1, 0].

    Returns:
        DataFrame with four rows of cholesterol data.
    """
    return pd.DataFrame({
        "told_high_cholesterol": [1.0, 0.0, np.nan, 1.0],
        "other_col": [1, 2, 3, 4],
    })


@pytest.fixture
def all_nan_cholesterol_dataframe() -> pd.DataFrame:
    """
    DataFrame where told_high_cholesterol is entirely NaN.

    Returns:
        DataFrame with three rows, all NaN in cholesterol column.
    """
    return pd.DataFrame({
        "told_high_cholesterol": [np.nan, np.nan, np.nan],
        "other_col": [1, 2, 3],
    })


@pytest.fixture
def no_nan_cholesterol_dataframe() -> pd.DataFrame:
    """
    DataFrame where told_high_cholesterol has no NaN values.

    Returns:
        DataFrame with four rows, no NaN in cholesterol column.
    """
    return pd.DataFrame({
        "told_high_cholesterol": [1.0, 0.0, 1.0, 0.0],
        "other_col": [1, 2, 3, 4],
    })


# ---------------------------------------------------------------------------
# ColumnSelector fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def post_fe_dataframe() -> pd.DataFrame:
    """
    DataFrame with all 13 SELECTED_FEATURES plus extra columns to be dropped.

    Simulates output of the FE transformers before column selection.
    Extra columns: middle_age, senior, age_unknown.

    Returns:
        DataFrame with 16 columns (13 selected + 3 extra).
    """
    return pd.DataFrame({
        "young_adult": [1, 0],
        "elderly": [0, 0],
        "waist_to_height_ratio": [0.5, 0.6],
        "told_high_cholesterol_missing": [0, 1],
        "told_high_bp": [0, 1],
        "told_high_cholesterol": [0.0, 1.0],
        "is_female": [1, 0],
        "drinking_frequency": [2, 3],
        "diastolic_bp": [78, 82],
        "systolic_bp": [120, 130],
        "education_level": [3, 4],
        "phq9_score": [2, 5],
        "vigorous_minutes_per_week": [60, 0],
        # Extra columns that should be dropped
        "middle_age": [0, 1],
        "senior": [0, 0],
        "age_unknown": [0, 0],
    })


@pytest.fixture
def shuffled_post_fe_dataframe(post_fe_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Same data as post_fe_dataframe but columns in reversed order.
    Used to verify that ColumnSelector enforces canonical column order
    regardless of input column order.

    Args:
        post_fe_dataframe: Base DataFrame with all required columns.

    Returns:
        DataFrame with columns in reversed order.
    """
    return post_fe_dataframe[list(reversed(post_fe_dataframe.columns.tolist()))]


# ---------------------------------------------------------------------------
# Imputer fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def imputer_dataframe_with_nans() -> pd.DataFrame:
    """
    DataFrame with all 13 selected features, NaN in specific positions.

    NaN positions:
    - vigorous_minutes_per_week[1]: zero-fill col → expected 0
    - drinking_frequency[2]: zero-fill col → expected 0
    - waist_to_height_ratio[1]: median-fill col → expected median([0.5, 0.7]) = 0.6

    Passthrough columns have no NaN.

    Returns:
        DataFrame with three rows and intentional NaN values.
    """
    return pd.DataFrame({
        "young_adult": [1, 0, 0],
        "elderly": [0, 0, 1],
        "waist_to_height_ratio": [0.5, np.nan, 0.7],
        "told_high_cholesterol_missing": [0, 1, 0],
        "told_high_bp": [1.0, 0.0, 1.0],
        "told_high_cholesterol": [1.0, 0.0, 0.0],
        "is_female": [0.0, 1.0, 1.0],
        "drinking_frequency": [2.0, 3.0, np.nan],
        "diastolic_bp": [80.0, 75.0, 85.0],
        "systolic_bp": [120.0, 125.0, 130.0],
        "education_level": [3.0, 4.0, 2.0],
        "phq9_score": [2.0, 5.0, 3.0],
        "vigorous_minutes_per_week": [60.0, np.nan, 30.0],
    })


@pytest.fixture
def imputer_dataframe_no_nans() -> pd.DataFrame:
    """
    DataFrame with all 13 selected features and no NaN values.

    Returns:
        DataFrame with two rows and no missing values.
    """
    return pd.DataFrame({
        "young_adult": [1, 0],
        "elderly": [0, 1],
        "waist_to_height_ratio": [0.5, 0.6],
        "told_high_cholesterol_missing": [0, 0],
        "told_high_bp": [1.0, 0.0],
        "told_high_cholesterol": [1.0, 0.0],
        "is_female": [0.0, 1.0],
        "drinking_frequency": [2.0, 3.0],
        "diastolic_bp": [80.0, 75.0],
        "systolic_bp": [120.0, 125.0],
        "education_level": [3.0, 4.0],
        "phq9_score": [2.0, 5.0],
        "vigorous_minutes_per_week": [60.0, 30.0],
    })


@pytest.fixture
def imputer_fit_dataframe() -> pd.DataFrame:
    """
    DataFrame used to fit the Imputer with known medians.
    diastolic_bp values [60, 80, 100] → median = 80.
    Used in test_fit_transform_separation to verify the Imputer uses
    fit-data medians when transforming separate data.

    Returns:
        DataFrame with three rows and no NaN values.
    """
    return pd.DataFrame({
        "young_adult": [1, 0, 0],
        "elderly": [0, 0, 1],
        "waist_to_height_ratio": [0.5, 0.6, 0.7],
        "told_high_cholesterol_missing": [0, 1, 0],
        "told_high_bp": [1.0, 0.0, 1.0],
        "told_high_cholesterol": [1.0, 0.0, 0.0],
        "is_female": [0.0, 1.0, 1.0],
        "drinking_frequency": [2.0, 3.0, 1.0],
        "diastolic_bp": [60.0, 80.0, 100.0],
        "systolic_bp": [120.0, 125.0, 130.0],
        "education_level": [3.0, 4.0, 2.0],
        "phq9_score": [2.0, 5.0, 3.0],
        "vigorous_minutes_per_week": [60.0, 30.0, 0.0],
    })


@pytest.fixture
def imputer_transform_dataframe() -> pd.DataFrame:
    """
    DataFrame to be transformed after fitting on imputer_fit_dataframe.
    diastolic_bp[0] is NaN; after transform it should be 80 (the fit median),
    not 200 (the median of this DataFrame's non-NaN values).

    Returns:
        DataFrame with three rows where diastolic_bp[0] is NaN.
    """
    return pd.DataFrame({
        "young_adult": [1, 0, 0],
        "elderly": [0, 1, 0],
        "waist_to_height_ratio": [0.5, 0.6, 0.7],
        "told_high_cholesterol_missing": [0, 0, 1],
        "told_high_bp": [1.0, 0.0, 1.0],
        "told_high_cholesterol": [1.0, 0.0, 0.0],
        "is_female": [0.0, 1.0, 0.0],
        "drinking_frequency": [2.0, 3.0, 1.0],
        "diastolic_bp": [np.nan, 200.0, 200.0],
        "systolic_bp": [120.0, 125.0, 130.0],
        "education_level": [3.0, 4.0, 2.0],
        "phq9_score": [2.0, 5.0, 3.0],
        "vigorous_minutes_per_week": [60.0, 30.0, 0.0],
    })
