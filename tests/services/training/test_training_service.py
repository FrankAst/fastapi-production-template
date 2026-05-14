from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from app.services.exceptions import ArtifactPersistError
from app.services.processing.column_selector import SELECTED_FEATURES
from app.services.training import TrainingResult, TrainingService


def test_train_returns_correct_counts(
    training_service: TrainingService,
    nhanes_training_dataframe: pd.DataFrame,
) -> None:
    n_samples = len(nhanes_training_dataframe)
    n_test = round(n_samples * training_service.lr_config.test_size)
    n_train = n_samples - n_test

    result = training_service.train(nhanes_training_dataframe)

    assert isinstance(result, TrainingResult)
    assert result.n_samples == n_samples
    assert result.n_train == n_train
    assert result.n_test == n_test
    assert result.n_bootstrap == training_service.lr_config.n_bootstrap_pred
    assert result.n_shap_background == training_service.lr_config.n_shap_background


def test_train_writes_artifact_files(
    training_service: TrainingService,
    nhanes_training_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    training_service.train(nhanes_training_dataframe)

    assert (tmp_path / "eval_model.joblib").exists()
    assert (tmp_path / "production_model.joblib").exists()
    assert (tmp_path / "bootstrap_ensemble.joblib").exists()
    assert (tmp_path / "test_set.joblib").exists()
    assert (tmp_path / "shap_background.joblib").exists()


def test_persisted_shap_background_has_expected_shape_and_dtype(
    training_service: TrainingService,
    nhanes_training_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    training_service.train(nhanes_training_dataframe)

    background = joblib.load(tmp_path / "shap_background.joblib")
    production_model = joblib.load(tmp_path / "production_model.joblib")
    classifier_feature_order = list(production_model[-1].feature_names_in_)

    assert isinstance(background, pd.DataFrame)
    assert background.shape == (
        training_service.lr_config.n_shap_background,
        len(SELECTED_FEATURES),
    )
    assert list(background.columns) == classifier_feature_order
    assert all(np.issubdtype(dtype, np.number) for dtype in background.dtypes)
    assert not background.isna().any().any()


def test_train_raises_artifact_persist_error_when_shap_background_save_fails(
    training_service: TrainingService,
    nhanes_training_dataframe: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shap_background_path = tmp_path / "shap_background.joblib"
    real_dump = joblib.dump

    def dump_failing_on_shap_background(value: object, filename: Path) -> object:
        if filename == shap_background_path:
            msg = "write failed"
            raise OSError(msg)
        return real_dump(value, filename)

    monkeypatch.setattr(
        "app.services.helper.joblib.dump", dump_failing_on_shap_background
    )

    with pytest.raises(ArtifactPersistError) as exc_info:
        training_service.train(nhanes_training_dataframe)

    assert exc_info.value.path == shap_background_path
