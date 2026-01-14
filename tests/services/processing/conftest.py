"""Fixtures for preprocessing tests."""

from typing import Any

import pandas as pd
import pytest


@pytest.fixture
def valid_training_data() -> dict[str, list[Any]]:
    """
    Returns:
        Dictionary with feature columns and target values.
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
    Returns:
        DataFrame with feature columns and target values.
    """
    return pd.DataFrame(valid_training_data)


@pytest.fixture
def df_age_binning_test_cases() -> pd.DataFrame:
    """
    Comprehensive DataFrame with all age test cases.

    Note: Dataset only contains adults (18+).

    Age semantics (from service):
    - 18-44  -> young_adult
    - 45-64  -> middle_age
    - 65-79  -> senior
    - 80     -> elderly (top-coded age)
    - NaN    -> age_unknown

    Rows are organized by category for easy slicing:
    - Rows 0-3: Young adult [18, 45) -> ages 18, 25, 30, 44
    - Rows 4-7: Middle age [45, 65) -> ages 45, 50, 55, 64
    - Rows 8-11: Senior [65, 80) -> ages 65, 70, 75, 79
    - Rows 12-13: Elderly (top-coded) -> age 80
    - Rows 14-15: Unknown -> NaN values

    Returns:
        DataFrame with RIDAGEYR and feature columns.
    """
    return pd.DataFrame({
        "RIDAGEYR": [
            # Young adult [18, 45)
            18,
            25,
            30,
            44,
            # Middle age [45, 65)
            45,
            50,
            55,
            64,
            # Senior [65, 80)
            65,
            70,
            75,
            79,
            # Elderly (top-coded age = 80)
            80,
            80,
            # Unknown
            None,
            None,
        ],
        "feature": list(range(1, 17)),
    })
