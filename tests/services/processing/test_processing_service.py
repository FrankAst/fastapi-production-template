"""Tests for AgeBinner transformer."""

import pandas as pd

from app.services.processing import AgeBinner

# =============================================================================
# Age Binning Column Tests
# =============================================================================


def test_creates_expected_columns(age_categories: list[str]) -> None:
    """
    Test that age binning creates the expected one-hot encoded columns.

    Args:
        age_categories: List of expected age category column names.
    """
    df = pd.DataFrame({
        "RIDAGEYR": [25, 50, 70, 80, None],
        "feature": [1, 2, 3, 4, 5],
    })
    result = AgeBinner().transform(df)

    assert set(age_categories).issubset(result.columns)


def test_removes_original_column() -> None:
    """Test that RIDAGEYR column is removed after binning."""
    df = pd.DataFrame({"RIDAGEYR": [25], "feature": [1]})
    result = AgeBinner().transform(df)

    assert "RIDAGEYR" not in result.columns


# =============================================================================
# Age Binning Category Tests
# =============================================================================


def test_age_maps_to_correct_category(
    age_with_expected_category: tuple[float | None, str],
    age_categories: list[str],
) -> None:
    """
    Test that each age maps to exactly one correct category.

    Args:
        age_with_expected_category: Tuple of (age, expected_category).
        age_categories: List of all age category column names.
    """
    age, expected_category = age_with_expected_category
    df = pd.DataFrame({"RIDAGEYR": [age], "feature": [1]})
    result = AgeBinner().transform(df)

    for category in age_categories:
        expected = 1 if category == expected_category else 0
        actual = result[category].iloc[0]
        assert actual == expected


def test_mutual_exclusivity(
    valid_training_dataframe: pd.DataFrame,
    age_categories: list[str],
) -> None:
    """
    Test that each row has exactly one age category set to 1.

    Args:
        valid_training_dataframe: Base DataFrame with training data.
        age_categories: List of all age category column names.
    """
    df = valid_training_dataframe.copy()
    df["RIDAGEYR"] = [18, 45, 65, 80]
    result = AgeBinner().transform(df)

    row_sums = result[age_categories].sum(axis=1)
    assert all(row_sums == 1), "Each row should have exactly one age category"
