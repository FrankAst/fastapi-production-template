"""API-level handler contract tests.

For each registered exception, assert that FastAPI converts it into the
documented (status_code, ErrorResponse body) pair. Route tests in
`tests/api/routes/` cover happy-path behavior; this module pins the
error-path contract.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandera as pa
import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pandas import DataFrame
from pandera.errors import SchemaErrors

from app.injections import configure_container
from app.services import ArtifactPersistError
from app.services.evaluation import EvaluationService, NoEvaluationArtifactsError
from app.services.prediction import (
    NoBootstrapEnsembleError,
    NoShapBackgroundError,
    NoTrainedModelError,
    PredictionService,
)
from app.services.processing import MissingFeatureColumnsError
from app.services.training import TrainingService
from app.utils import CsvContentError, CsvFormatError, CsvSizeError

_TO_VALIDATED_DATAFRAME = (
    "app.api.routes.prediction.schemas.SinglePredictionRequest.to_validated_dataframe"
)
_PARSE_TRAINING_UPLOAD = "app.api.routes.train.endpoints.parse_training_upload"
_SCHEMA_PATH = (
    "app.domain.schema_validator.SchemaValidator.get_training_input_schema_path"
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


@pytest.mark.parametrize(
    "exception_type",
    [
        pytest.param(NoTrainedModelError, id="no_trained_model"),
        pytest.param(NoBootstrapEnsembleError, id="no_bootstrap_ensemble"),
        pytest.param(NoShapBackgroundError, id="no_shap_background"),
    ],
)
def test_prediction_artifact_missing_returns_409(
    client: TestClient,
    realistic_payload: dict[str, object],
    exception_type: type[Exception],
) -> None:
    container = configure_container()
    failing_service = MagicMock(spec=PredictionService)
    failing_service.predict.side_effect = exception_type()

    with (
        patch(_TO_VALIDATED_DATAFRAME, return_value=DataFrame([realistic_payload])),
        container.prediction_service.override(failing_service),
    ):
        response = client.post("/prediction/single", json=realistic_payload)

    assert response.status_code == status.HTTP_409_CONFLICT
    body = response.json()
    assert isinstance(body["detail"], str)
    assert len(body["detail"]) > 0
    assert "failureCases" not in body


def test_no_evaluation_artifacts_returns_409(client: TestClient) -> None:
    container = configure_container()
    failing_service = MagicMock(spec=EvaluationService)
    failing_service.evaluate.side_effect = NoEvaluationArtifactsError

    with container.evaluation_service.override(failing_service):
        response = client.get("/evaluate/")

    assert response.status_code == status.HTTP_409_CONFLICT
    body = response.json()
    assert isinstance(body["detail"], str)
    assert len(body["detail"]) > 0
    assert "failureCases" not in body


def test_no_training_schema_returns_409(
    client: TestClient,
    realistic_payload: dict[str, object],
    tmp_path: Path,
) -> None:
    nonexistent_schema = tmp_path / "does_not_exist.yaml"

    with patch(_SCHEMA_PATH, return_value=nonexistent_schema):
        response = client.post("/prediction/single", json=realistic_payload)

    assert response.status_code == status.HTTP_409_CONFLICT
    body = response.json()
    assert isinstance(body["detail"], str)
    assert "training schema" in body["detail"].lower()
    assert "failureCases" not in body


def test_artifact_persist_returns_500_with_sanitized_detail(client: TestClient) -> None:
    container = configure_container()
    leaky_path = Path("/secret/server/path/model.joblib")
    failing_service = MagicMock(spec=TrainingService)
    failing_service.train.side_effect = ArtifactPersistError(leaky_path)

    with (
        patch(_PARSE_TRAINING_UPLOAD, new_callable=AsyncMock, return_value=DataFrame()),
        container.training_service.override(failing_service),
    ):
        response = client.post("/train/", files={"file": ("data.csv", b"", "text/csv")})

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    body = response.json()
    assert isinstance(body["detail"], str)
    assert "/secret/server/path" not in body["detail"]
    assert "failureCases" not in body


def test_missing_feature_columns_returns_500(
    client: TestClient,
    realistic_payload: dict[str, object],
) -> None:
    container = configure_container()
    missing = ["BMXWAIST", "BMXHT"]
    failing_service = MagicMock(spec=PredictionService)
    failing_service.predict.side_effect = MissingFeatureColumnsError(missing)

    with (
        patch(_TO_VALIDATED_DATAFRAME, return_value=DataFrame([realistic_payload])),
        container.prediction_service.override(failing_service),
    ):
        response = client.post("/prediction/single", json=realistic_payload)

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    body = response.json()
    assert isinstance(body["detail"], str)
    assert all(col in body["detail"] for col in missing)
    assert "failureCases" not in body


@pytest.mark.parametrize(
    ("exception", "expected_status"),
    [
        pytest.param(
            CsvFormatError("Invalid file type. Filename is missing."),
            status.HTTP_400_BAD_REQUEST,
            id="csv_format",
        ),
        pytest.param(
            CsvSizeError("File too large. Maximum size is 50 MB."),
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            id="csv_size",
        ),
        pytest.param(
            CsvContentError("CSV contains no data rows."),
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            id="csv_content",
        ),
    ],
)
def test_csv_errors_return_documented_status(
    client: TestClient,
    exception: Exception,
    expected_status: int,
) -> None:
    with patch(
        _PARSE_TRAINING_UPLOAD,
        new_callable=AsyncMock,
        side_effect=exception,
    ):
        response = client.post("/train/", files={"file": ("data.csv", b"", "text/csv")})

    assert response.status_code == expected_status
    body = response.json()
    assert isinstance(body["detail"], str)
    assert len(body["detail"]) > 0
    assert "failureCases" not in body


def test_request_validation_error_returns_422_with_failure_cases(
    client: TestClient,
) -> None:
    response = client.post("/prediction/single", json={"incomplete": "body"})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    body = response.json()
    assert body["detail"] == "Request body validation failed"
    assert isinstance(body["failureCases"], list)
    assert len(body["failureCases"]) > 0
    first_case = body["failureCases"][0]
    assert "loc" in first_case
    assert "msg" in first_case
    assert "type" in first_case


def test_schema_errors_returns_422_with_failure_cases(
    client: TestClient,
    realistic_payload: dict[str, object],
) -> None:
    out_of_domain_payload = {**realistic_payload, "RIDAGEYR": -1.0}
    age_schema = pa.DataFrameSchema({
        "RIDAGEYR": pa.Column(float, checks=pa.Check.ge(0))
    })
    with pytest.raises(SchemaErrors) as captured:
        age_schema.validate(DataFrame([out_of_domain_payload]), lazy=True)

    with patch(_TO_VALIDATED_DATAFRAME, side_effect=captured.value):
        response = client.post("/prediction/single", json=realistic_payload)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    body = response.json()
    assert body["detail"] == "Data validation failed"
    assert isinstance(body["failureCases"], list)
    assert len(body["failureCases"]) > 0
