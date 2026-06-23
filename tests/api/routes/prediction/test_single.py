import itertools
from unittest.mock import MagicMock, patch

import pandera as pa
import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pandas import DataFrame
from pandera.errors import SchemaErrors

from app.domain import (
    PredictionOutput,
    ShapContribution,
    ShapExplanation,
)
from app.injections import configure_container
from app.services.processing.column_selector import SELECTED_FEATURES

_PATCH_TARGET = (
    "app.api.routes.prediction.schemas.SinglePredictionRequest.to_validated_dataframe"
)


@pytest.fixture
def realistic_payload() -> dict[str, object]:
    return {
        "RIDAGEYR": 45,
        "BMXWAIST": 95.0,
        "BMXHT": 170.0,
        "toldHighBp": True,
        "toldHighCholesterol": False,
        "isFemale": True,
        "drinkingFrequency": 2,
        "diastolicBp": 80.0,
        "systolicBp": 120.0,
        "educationLevel": 4,
        "phq9Score": 5,
        "vigorousMinutesPerWeek": 30,
    }


@pytest.fixture
def known_prediction_output() -> PredictionOutput:
    contributions = tuple(
        ShapContribution(feature=feature, shap_value=(13 - index) * 0.1)
        for index, feature in enumerate(SELECTED_FEATURES)
    )
    base_value = -0.5
    final_value = base_value + sum(c.shap_value for c in contributions)
    return PredictionOutput(
        probability=0.62,
        ci_lower=0.55,
        ci_upper=0.69,
        is_positive=True,
        shap=ShapExplanation(
            base_value=base_value,
            contributions=contributions,
            final_value=final_value,
        ),
    )


@pytest.fixture
def mock_prediction_service(
    known_prediction_output: PredictionOutput,
) -> MagicMock:
    service = MagicMock()
    service.predict.return_value = known_prediction_output
    return service


def test_single_prediction_returns_full_response(
    client: TestClient,
    realistic_payload: dict[str, object],
    mock_prediction_service: MagicMock,
    known_prediction_output: PredictionOutput,
) -> None:
    container = configure_container()

    with (
        patch(_PATCH_TARGET, return_value=DataFrame([realistic_payload])),
        container.prediction_service.override(mock_prediction_service),
    ):
        response = client.post("/prediction/single", json=realistic_payload)

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    expected_prob = pytest.approx(known_prediction_output.probability)  # pyright: ignore[reportUnknownMemberType]
    expected_ci_lower = pytest.approx(known_prediction_output.ci_lower)  # pyright: ignore[reportUnknownMemberType]
    expected_ci_upper = pytest.approx(known_prediction_output.ci_upper)  # pyright: ignore[reportUnknownMemberType]
    assert body["probability"] == expected_prob
    assert body["ciLower"] == expected_ci_lower
    assert body["ciUpper"] == expected_ci_upper
    assert body["isPositive"] is known_prediction_output.is_positive
    assert body["ciLower"] <= body["probability"] <= body["ciUpper"]
    assert len(body["shap"]["contributions"]) == len(SELECTED_FEATURES)
    abs_values = [abs(c["shapValue"]) for c in body["shap"]["contributions"]]
    for previous, current in itertools.pairwise(abs_values):
        assert previous >= current


def test_single_prediction_preserves_shap_additive_invariant(
    client: TestClient,
    realistic_payload: dict[str, object],
    mock_prediction_service: MagicMock,
) -> None:
    container = configure_container()

    with (
        patch(_PATCH_TARGET, return_value=DataFrame([realistic_payload])),
        container.prediction_service.override(mock_prediction_service),
    ):
        response = client.post("/prediction/single", json=realistic_payload)

    body = response.json()
    base = body["shap"]["baseValue"]
    contributions_sum = sum(c["shapValue"] for c in body["shap"]["contributions"])
    expected_final = pytest.approx(base + contributions_sum)  # pyright: ignore[reportUnknownMemberType]

    assert body["shap"]["finalValue"] == expected_final


def test_single_prediction_rejects_payload_missing_a_required_field(
    client: TestClient,
    realistic_payload: dict[str, object],
) -> None:
    incomplete_payload = {
        key: value for key, value in realistic_payload.items() if key != "RIDAGEYR"
    }

    response = client.post("/prediction/single", json=incomplete_payload)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_single_prediction_rejects_out_of_domain_value(
    client: TestClient,
    realistic_payload: dict[str, object],
) -> None:
    out_of_domain_payload = {**realistic_payload, "RIDAGEYR": -1.0}
    age_schema = pa.DataFrameSchema({
        "RIDAGEYR": pa.Column(float, checks=pa.Check.ge(0))
    })
    with pytest.raises(SchemaErrors) as captured:
        age_schema.validate(DataFrame([out_of_domain_payload]), lazy=True)

    with patch(_PATCH_TARGET, side_effect=captured.value):
        response = client.post("/prediction/single", json=realistic_payload)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.parametrize(
    "nullable_field",
    [
        "BMXWAIST",
        "BMXHT",
        "toldHighBp",
        "toldHighCholesterol",
        "drinkingFrequency",
        "diastolicBp",
        "systolicBp",
        "phq9Score",
        "vigorousMinutesPerWeek",
    ],
)
def test_single_prediction_accepts_null_for_nullable_fields(
    client: TestClient,
    realistic_payload: dict[str, object],
    mock_prediction_service: MagicMock,
    nullable_field: str,
) -> None:
    payload_with_null = {**realistic_payload, nullable_field: None}
    container = configure_container()

    with (
        patch(_PATCH_TARGET, return_value=DataFrame([payload_with_null])),
        container.prediction_service.override(mock_prediction_service),
    ):
        response = client.post("/prediction/single", json=payload_with_null)

    assert response.status_code == status.HTTP_200_OK


def test_single_prediction_rejects_null_education_level(
    client: TestClient,
    realistic_payload: dict[str, object],
) -> None:
    payload_with_null_education = {**realistic_payload, "educationLevel": None}

    response = client.post("/prediction/single", json=payload_with_null_education)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_single_prediction_rejects_clinically_inconsistent_payload(
    client: TestClient,
    realistic_payload: dict[str, object],
) -> None:
    inconsistent_payload = {
        **realistic_payload,
        "systolicBp": 70.0,
        "diastolicBp": 90.0,
    }

    response = client.post("/prediction/single", json=inconsistent_payload)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    body = response.json()
    assert body["detail"] == (
        "Input contains rows with physiologically inconsistent feature combinations"
    )
    assert body["failureCases"] == [
        {"column": "systolic_bp", "check": "greater_than(diastolic_bp)", "index": 0}
    ]
