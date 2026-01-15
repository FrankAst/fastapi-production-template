"""Fixtures for preprocessing tests."""

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# Base Training Data Fixtures
# =============================================================================


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


# =============================================================================
# Age Binner Fixtures
# =============================================================================


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
    return [
        pytest.param((18, "young_adult"), id="young_adult_lower"),
        pytest.param((30, "young_adult"), id="young_adult_mid"),
        pytest.param((44, "young_adult"), id="young_adult_upper"),
        pytest.param((45, "middle_age"), id="middle_age_lower"),
        pytest.param((55, "middle_age"), id="middle_age_mid"),
        pytest.param((64, "middle_age"), id="middle_age_upper"),
        pytest.param((65, "senior"), id="senior_lower"),
        pytest.param((72, "senior"), id="senior_mid"),
        pytest.param((79, "senior"), id="senior_upper"),
        pytest.param((80, "elderly"), id="elderly_top_coded"),
        pytest.param((np.nan, "age_unknown"), id="unknown_nan"),
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
