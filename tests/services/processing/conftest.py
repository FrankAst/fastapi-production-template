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
