from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pandas import DataFrame

from app.domain.models import LRVifBicConfig
from app.injections import configure_container
from app.services.training import TrainingResult


@pytest.fixture
def mock_training_service() -> MagicMock:
    """Mock TrainingService with a stubbed train() method.

    Returns:
        MagicMock configured to return a known TrainingResult.
    """
    service = MagicMock()
    service.train.return_value = TrainingResult(
        n_samples=10,
        n_train=8,
        n_test=2,
        n_bootstrap=2,
        n_shap_background=2,
        threshold=LRVifBicConfig().threshold,
    )
    return service


def test_train_endpoint_returns_summary(
    client: TestClient,
    mock_training_service: MagicMock,
) -> None:
    container = configure_container()

    with (
        patch(
            "app.api.routes.train.endpoints.parse_training_upload",
            new_callable=AsyncMock,
            return_value=DataFrame(),
        ),
        container.training_service.override(mock_training_service),
    ):
        response = client.post("/train/", files={"file": ("data.csv", b"", "text/csv")})

    config = LRVifBicConfig()
    known_result = mock_training_service.train.return_value

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["nSamples"] == known_result.n_samples
    assert body["nTrain"] == known_result.n_train
    assert body["nTest"] == known_result.n_test
    assert body["nBootstrap"] == known_result.n_bootstrap
    assert body["nShapBackground"] == known_result.n_shap_background
    assert body["threshold"] == config.threshold
