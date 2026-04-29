import itertools
from pathlib import Path

import numpy as np
import pytest

from app.domain import BatchPredictionOutput, LRVifBicConfig, PredictionInput
from app.services.prediction import (
    NoBootstrapEnsembleError,
    NoShapBackgroundError,
    NoTrainedModelError,
    PredictionService,
)
from app.services.processing.column_selector import SELECTED_FEATURES


def test_predict_returns_populated_prediction_output(
    prediction_service: PredictionService,
    known_input_row: PredictionInput,
) -> None:
    output = prediction_service.predict(known_input_row)

    assert 0 < output.probability < 1
    assert output.ci_lower <= output.ci_upper
    assert len(output.shap.contributions) == len(SELECTED_FEATURES)
    expected_flag = output.probability > prediction_service.lr_config.threshold
    assert output.is_positive is expected_flag


def test_shap_contributions_sum_to_final_value(
    prediction_service: PredictionService,
    known_input_row: PredictionInput,
) -> None:
    output = prediction_service.predict(known_input_row)

    expected = output.shap.base_value + sum(
        contribution.shap_value for contribution in output.shap.contributions
    )

    assert output.shap.final_value == pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]


def test_sigmoid_of_shap_final_value_matches_probability(
    prediction_service: PredictionService,
    known_input_row: PredictionInput,
) -> None:
    output = prediction_service.predict(known_input_row)

    sigmoid = 1.0 / (1.0 + np.exp(-output.shap.final_value))

    assert sigmoid == pytest.approx(output.probability, rel=1e-6)  # pyright: ignore[reportUnknownMemberType]


def test_contributions_are_sorted_by_absolute_shap_value_descending(
    prediction_service: PredictionService,
    known_input_row: PredictionInput,
) -> None:
    output = prediction_service.predict(known_input_row)

    abs_values = [abs(c.shap_value) for c in output.shap.contributions]

    for previous, current in itertools.pairwise(abs_values):
        assert previous >= current


def test_contributions_are_a_tuple(
    prediction_service: PredictionService,
    known_input_row: PredictionInput,
) -> None:
    output = prediction_service.predict(known_input_row)

    assert isinstance(output.shap.contributions, tuple)


def test_batch_predict_returns_batch_prediction_output(
    prediction_service: PredictionService,
    known_input_row: PredictionInput,
) -> None:
    output = prediction_service.batch_predict(known_input_row)

    assert isinstance(output, BatchPredictionOutput)
    assert output.count == len(known_input_row.features)
    assert len(output.predictions) == output.count


def test_predict_raises_when_production_model_is_missing(
    fast_lr_config: LRVifBicConfig,
    known_input_row: PredictionInput,
    tmp_path: Path,
) -> None:
    service = PredictionService(
        lr_config=fast_lr_config,
        model_path=tmp_path / "missing.joblib",
    )

    with pytest.raises(NoTrainedModelError):
        service.predict(known_input_row)


def test_predict_raises_when_bootstrap_ensemble_is_missing(
    fast_lr_config: LRVifBicConfig,
    known_input_row: PredictionInput,
    trained_artifacts: None,  # noqa: ARG001  # pylint: disable=unused-argument
    tmp_path: Path,
) -> None:
    service = PredictionService(
        lr_config=fast_lr_config,
        bootstrap_path=tmp_path / "missing.joblib",
    )

    with pytest.raises(NoBootstrapEnsembleError):
        service.predict(known_input_row)


def test_predict_raises_when_shap_background_is_missing(
    fast_lr_config: LRVifBicConfig,
    known_input_row: PredictionInput,
    trained_artifacts: None,  # noqa: ARG001  # pylint: disable=unused-argument
    tmp_path: Path,
) -> None:
    service = PredictionService(
        lr_config=fast_lr_config,
        shap_background_path=tmp_path / "missing.joblib",
    )

    with pytest.raises(NoShapBackgroundError):
        service.predict(known_input_row)
