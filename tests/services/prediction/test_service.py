import itertools
from pathlib import Path

import numpy as np
import pandas as pd
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


def test_contribution_features_match_selected_features(
    prediction_service: PredictionService,
    known_input_row: PredictionInput,
) -> None:
    output = prediction_service.predict(known_input_row)

    feature_names = {c.feature for c in output.shap.contributions}

    assert feature_names == set(SELECTED_FEATURES)


@pytest.mark.parametrize(
    ("raw_column", "perturbed_value", "expected_feature"),
    [
        pytest.param("told_high_bp", 1.0, "told_high_bp", id="told_high_bp"),
        pytest.param(
            "told_high_cholesterol",
            1.0,
            "told_high_cholesterol",
            id="told_high_cholesterol",
        ),
        pytest.param("is_female", 1.0, "is_female", id="is_female"),
        pytest.param(
            "drinking_frequency", 4.0, "drinking_frequency", id="drinking_frequency"
        ),
        pytest.param("diastolic_bp", 130.0, "diastolic_bp", id="diastolic_bp"),
        pytest.param("systolic_bp", 200.0, "systolic_bp", id="systolic_bp"),
        pytest.param("education_level", 5.0, "education_level", id="education_level"),
        pytest.param("phq9_score", 27.0, "phq9_score", id="phq9_score"),
        pytest.param(
            "vigorous_minutes_per_week",
            1500.0,
            "vigorous_minutes_per_week",
            id="vigorous_minutes_per_week",
        ),
        pytest.param("BMXWAIST", 130.0, "waist_to_height_ratio", id="waist_ratio"),
    ],
)
def test_perturbing_one_feature_dominates_its_shap_change(
    prediction_service: PredictionService,
    baseline_patient_features: pd.DataFrame,
    raw_column: str,
    perturbed_value: float,
    expected_feature: str,
) -> None:
    baseline_output = prediction_service.predict(
        PredictionInput(features=baseline_patient_features)
    )
    perturbed_features = baseline_patient_features.copy()
    perturbed_features[raw_column] = perturbed_value
    perturbed_output = prediction_service.predict(
        PredictionInput(features=perturbed_features)
    )

    baseline_by_name = {
        c.feature: c.shap_value for c in baseline_output.shap.contributions
    }
    perturbed_by_name = {
        c.feature: c.shap_value for c in perturbed_output.shap.contributions
    }
    deltas = {
        name: abs(perturbed_by_name[name] - baseline_by_name[name])
        for name in baseline_by_name
    }

    dominant_feature = max(deltas, key=lambda name: deltas[name])

    assert dominant_feature == expected_feature


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
