# ruff: noqa: DOC201
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from app.services.processing.scaler.config import CONTINUOUS_COLS


@pytest.fixture
def valid_training_data() -> dict[str, list[Any]]:
    """Raw training data as a dictionary."""
    return {
        "RIDAGEYR": [25, 30, 35, 40],
        "income": [50000, 60000, 70000, 80000],
        "education_years": [12, 16, 18, 20],
        "target": [0, 1, 1, 0],
    }


@pytest.fixture
def valid_training_dataframe(valid_training_data: dict[str, list[Any]]) -> pd.DataFrame:
    """Training DataFrame built from valid_training_data."""
    return pd.DataFrame(valid_training_data)


@pytest.fixture
def dataframe_with_all_age_categories(
    valid_training_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Base training DataFrame with RIDAGEYR covering all four age bins."""
    return valid_training_dataframe.assign(RIDAGEYR=[18, 45, 65, 80])


@pytest.fixture
def age_categories() -> list[str]:
    """All one-hot column names produced by AgeBinner."""
    return ["young_adult", "middle_age", "senior", "elderly", "age_unknown"]


def _age_test_cases() -> list[Any]:
    """(age, category, id) tuples covering all bin boundaries and edge cases."""
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
def _age_case(request: pytest.FixtureRequest) -> tuple[float | None, str]:
    return cast("tuple[float | None, str]", request.param)


@pytest.fixture
def age_dataframe(_age_case: tuple[float | None, str]) -> pd.DataFrame:
    """Single-row DataFrame with the parametrized RIDAGEYR value."""
    age, _ = _age_case
    return pd.DataFrame({"RIDAGEYR": [age], "feature": [1]})


@pytest.fixture
def expected_age_category(_age_case: tuple[float | None, str]) -> str:
    """The AgeBinner category column expected to be 1 for the parametrized age."""
    return _age_case[1]


@pytest.fixture
def waist_height_dataframe() -> pd.DataFrame:
    """Two-row DataFrame: row 0 ratio=0.5 (90/180), row 1 ratio=0.6 (96/160)."""
    return pd.DataFrame({
        "BMXWAIST": [90.0, 96.0],
        "BMXHT": [180.0, 160.0],
        "other_col": [1, 2],
    })


def _waist_height_edge_cases() -> list[Any]:
    """(waist, height, expected_ratio) edge cases including zero and NaN inputs."""
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
    """Parametrized (waist, height, expected_ratio) tuples for edge-case testing."""
    return cast("tuple[float | None, float | None, float | None]", request.param)


@pytest.fixture
def cholesterol_dataframe() -> pd.DataFrame:
    """told_high_cholesterol=[1, 0, NaN, 1] — NaN[2] triggers missingness flag."""
    return pd.DataFrame({
        "told_high_cholesterol": [1.0, 0.0, np.nan, 1.0],
        "other_col": [1, 2, 3, 4],
    })


@pytest.fixture
def all_nan_cholesterol_dataframe() -> pd.DataFrame:
    """told_high_cholesterol all NaN — every row triggers the missingness flag."""
    return pd.DataFrame({
        "told_high_cholesterol": [np.nan, np.nan, np.nan],
        "other_col": [1, 2, 3],
    })


@pytest.fixture
def no_nan_cholesterol_dataframe() -> pd.DataFrame:
    """told_high_cholesterol has no NaN — missingness flag should be all zeros."""
    return pd.DataFrame({
        "told_high_cholesterol": [1.0, 0.0, 1.0, 0.0],
        "other_col": [1, 2, 3, 4],
    })


@pytest.fixture
def post_fe_dataframe() -> pd.DataFrame:
    """All 13 SELECTED_FEATURES plus middle_age, senior, age_unknown as extras."""
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
        "middle_age": [0, 1],
        "senior": [0, 0],
        "age_unknown": [0, 0],
    })


@pytest.fixture
def shuffled_post_fe_dataframe(post_fe_dataframe: pd.DataFrame) -> pd.DataFrame:
    """post_fe_dataframe with columns reversed to verify canonical ordering."""
    return post_fe_dataframe[list(reversed(post_fe_dataframe.columns.tolist()))]


@pytest.fixture
def imputer_dataframe_with_nans() -> pd.DataFrame:
    """13 model features: NaN in vigorous_minutes[1], drinking_freq[2], whr[1]."""
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
    """13 model features with no NaN values."""
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
    """Fit DataFrame for Imputer: diastolic_bp=[60, 80, 100] → median=80."""
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
def imputer_transform_dataframe(imputer_fit_dataframe: pd.DataFrame) -> pd.DataFrame:
    """imputer_fit_dataframe with diastolic_bp[0]=NaN — fills to fit median (80)."""
    return imputer_fit_dataframe.assign(diastolic_bp=[np.nan, 200.0, 200.0])


@pytest.fixture(params=CONTINUOUS_COLS)
def continuous_col(request: pytest.FixtureRequest) -> str:
    """Parametrized fixture yielding each continuous column name in turn."""
    return str(request.param)


@pytest.fixture
def scaler_dataframe() -> pd.DataFrame:
    """13 model features, 4 rows, distinct continuous values, no NaN."""
    return pd.DataFrame({
        "young_adult": [1, 0, 1, 0],
        "elderly": [0, 0, 0, 1],
        "waist_to_height_ratio": [0.50, 0.55, 0.60, 0.52],
        "told_high_cholesterol_missing": [0, 0, 1, 0],
        "told_high_bp": [1, 0, 1, 0],
        "told_high_cholesterol": [1, 0, 0, 1],
        "is_female": [0, 1, 1, 0],
        "drinking_frequency": [2.0, 5.0, 0.0, 3.0],
        "diastolic_bp": [78.0, 90.0, 82.0, 75.0],
        "systolic_bp": [120.0, 140.0, 135.0, 125.0],
        "education_level": [3.0, 4.0, 2.0, 5.0],
        "phq9_score": [2.0, 8.0, 4.0, 1.0],
        "vigorous_minutes_per_week": [120.0, 0.0, 60.0, 30.0],
    })


@pytest.fixture
def scaler_fit_dataframe(imputer_fit_dataframe: pd.DataFrame) -> pd.DataFrame:
    """imputer_fit_dataframe with waist_to_height_ratio=[0.50, 0.55, 0.60]."""
    return imputer_fit_dataframe.assign(waist_to_height_ratio=[0.50, 0.55, 0.60])


@pytest.fixture
def scaler_transform_dataframe(scaler_fit_dataframe: pd.DataFrame) -> pd.DataFrame:
    """scaler_fit_dataframe with diastolic_bp=[80,80,80] — the fit mean → scaled=0.0."""
    return scaler_fit_dataframe.assign(diastolic_bp=[80.0, 80.0, 80.0])


@pytest.fixture
def pipeline_dataframe() -> pd.DataFrame:
    """Clean 12-column raw pipeline input covering all four AgeBinner age categories."""
    return pd.DataFrame({
        "RIDAGEYR": [25, 50, 70, 80],
        "BMXWAIST": [90.0, 100.0, 85.0, 95.0],
        "BMXHT": [170.0, 175.0, 165.0, 160.0],
        "told_high_cholesterol": [1.0, 0.0, 1.0, 0.0],
        "told_high_bp": [1.0, 0.0, 1.0, 0.0],
        "is_female": [0.0, 1.0, 0.0, 1.0],
        "drinking_frequency": [2.0, 5.0, 0.0, 3.0],
        "diastolic_bp": [78.0, 90.0, 82.0, 75.0],
        "systolic_bp": [120.0, 140.0, 135.0, 125.0],
        "education_level": [3.0, 4.0, 2.0, 5.0],
        "phq9_score": [2.0, 8.0, 4.0, 1.0],
        "vigorous_minutes_per_week": [120.0, 0.0, 60.0, 30.0],
    })


@pytest.fixture
def pipeline_dataframe_with_nans(pipeline_dataframe: pd.DataFrame) -> pd.DataFrame:
    """pipeline_dataframe with NaN injected to exercise all three imputer strategies."""
    return pipeline_dataframe.assign(
        told_high_cholesterol=[1.0, 0.0, np.nan, 0.0],
        diastolic_bp=[np.nan, 90.0, 82.0, 75.0],
        vigorous_minutes_per_week=[120.0, np.nan, 60.0, 30.0],
    )


@pytest.fixture
def pipeline_fit_dataframe(pipeline_dataframe: pd.DataFrame) -> pd.DataFrame:
    """pipeline_dataframe with diastolic_bp=[60, 80, 100, 80] → mean=80."""
    return pipeline_dataframe.assign(diastolic_bp=[60.0, 80.0, 100.0, 80.0])


@pytest.fixture
def pipeline_transform_dataframe(pipeline_dataframe: pd.DataFrame) -> pd.DataFrame:
    """pipeline_dataframe with diastolic_bp=[80,80,80,80] — fit mean → scaled=0.0."""
    return pipeline_dataframe.assign(diastolic_bp=[80.0, 80.0, 80.0, 80.0])
