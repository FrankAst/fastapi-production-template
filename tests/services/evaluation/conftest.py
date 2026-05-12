# pylint: disable=duplicate-code
from collections.abc import Generator
from pathlib import Path

import pandas as pd
import pytest

from app.domain.models import LRVifBicConfig
from app.services.evaluation import EvaluationService
from app.services.training import TrainingService
from app.settings import Settings


@pytest.fixture
def fast_lr_config() -> LRVifBicConfig:
    """LRVifBicConfig with fast bootstrap counts for training and evaluation.

    Returns:
        LRVifBicConfig with reduced bootstrap counts.
    """
    return LRVifBicConfig(n_bootstrap_pred=2, n_bootstrap_eval=10)


@pytest.fixture(autouse=True)
def mock_artifact_paths(tmp_path: Path) -> Generator[None]:
    """Redirect all artifact writes to tmp_path for every evaluation test."""
    paths: dict[str, Path] = {
        "EVAL_MODEL_PATH": tmp_path / "eval_model.joblib",
        "PRODUCTION_MODEL_PATH": tmp_path / "production_model.joblib",
        "BOOTSTRAP_ENSEMBLE_PATH": tmp_path / "bootstrap_ensemble.joblib",
        "TEST_SET_PATH": tmp_path / "test_set.joblib",
    }
    original = {attr: getattr(type(Settings), attr) for attr in paths}
    for attr, p in paths.items():
        setattr(type(Settings), attr, property(lambda _, val=p: val))  # type: ignore[misc]
    yield
    for attr, prop in original.items():
        setattr(type(Settings), attr, prop)


@pytest.fixture
def nhanes_training_dataframe() -> pd.DataFrame:
    """13-column NHANES DataFrame with 10 rows (5 per class) for stratified split.

    Returns:
        DataFrame with 10 rows and the expected NHANES columns.
    """
    columns = [
        "RIDAGEYR",
        "BMXWAIST",
        "BMXHT",
        "told_high_bp",
        "told_high_cholesterol",
        "is_female",
        "drinking_frequency",
        "diastolic_bp",
        "systolic_bp",
        "education_level",
        "phq9_score",
        "vigorous_minutes_per_week",
        "has_diabetes_or_prediabetes",
    ]
    rows = [
        [45.0, 95.0, 170.0, 1.0, 0.0, 1.0, 2.0, 80.0, 120.0, 4.0, 5.0, 0.0, 0.0],
        [62.0, 110.5, 165.0, 0.0, 1.0, 0.0, 0.0, 90.0, 145.0, 3.0, 12.0, 60.0, 1.0],
        [28.0, 78.0, 180.0, 0.0, 0.0, 1.0, 3.0, 70.0, 110.0, 5.0, 0.0, 150.0, 0.0],
        [55.0, 102.0, 168.0, 1.0, 1.0, 0.0, 1.0, 85.0, 135.0, 2.0, 8.0, 30.0, 1.0],
        [71.0, 88.0, 172.0, 0.0, 0.0, 1.0, 0.0, 75.0, 125.0, 4.0, 3.0, 0.0, 0.0],
        [33.0, 75.0, 175.0, 0.0, 0.0, 1.0, 4.0, 72.0, 115.0, 5.0, 0.0, 180.0, 0.0],
        [48.0, 99.0, 163.0, 1.0, 1.0, 0.0, 2.0, 88.0, 140.0, 3.0, 10.0, 45.0, 1.0],
        [66.0, 115.0, 169.0, 0.0, 0.0, 1.0, 0.0, 78.0, 118.0, 4.0, 2.0, 0.0, 0.0],
        [39.0, 84.0, 178.0, 0.0, 1.0, 0.0, 3.0, 68.0, 108.0, 5.0, 1.0, 120.0, 1.0],
        [58.0, 107.0, 166.0, 1.0, 0.0, 1.0, 1.0, 92.0, 138.0, 2.0, 7.0, 20.0, 1.0],
    ]
    return pd.DataFrame(rows, columns=columns)


@pytest.fixture
def evaluation_service(
    nhanes_training_dataframe: pd.DataFrame,
    fast_lr_config: LRVifBicConfig,
) -> EvaluationService:
    """EvaluationService with pre-built artifacts from a fast training run.

    Returns:
        EvaluationService ready to call evaluate().
    """
    TrainingService(lr_config=fast_lr_config).train(nhanes_training_dataframe)
    return EvaluationService(lr_config=fast_lr_config)
