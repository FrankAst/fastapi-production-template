from unittest.mock import MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.domain.models import LRVifBicConfig
from app.injections import configure_container
from app.services.evaluation import (
    EvaluationResult,
    EvaluationService,
    NoEvaluationArtifactsError,
)
from app.services.evaluation.evaluation_result import (
    ConfusionMatrixResult,
    MetricsResult,
    MetricWithCI,
)


@pytest.fixture
def known_result() -> EvaluationResult:
    """Fixed EvaluationResult for endpoint assertions.

    Returns:
        EvaluationResult with predictable values.
    """
    metric = MetricWithCI(value=0.5, ci_lower=0.4, ci_upper=0.6)
    return EvaluationResult(
        threshold=LRVifBicConfig().threshold,
        n_test=100,
        metrics=MetricsResult(
            average_precision=metric,
            precision=metric,
            recall=metric,
            f1=metric,
        ),
        confusion_matrix=ConfusionMatrixResult(tp=30, fp=20, tn=40, fn=10),
    )


@pytest.fixture
def mock_evaluation_service(known_result: EvaluationResult) -> MagicMock:
    """Mock EvaluationService returning a known result.

    Returns:
        MagicMock configured to return known_result from evaluate().
    """
    service = MagicMock(spec=EvaluationService)
    service.evaluate.return_value = known_result
    return service


def test_evaluate_endpoint_returns_summary(
    client: TestClient,
    mock_evaluation_service: MagicMock,
    known_result: EvaluationResult,
) -> None:
    container = configure_container()

    with container.evaluation_service.override(mock_evaluation_service):
        response = client.get("/evaluate/")

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["threshold"] == known_result.threshold
    assert body["nTest"] == known_result.n_test
    assert (
        body["metrics"]["averagePrecision"]["value"]
        == known_result.metrics.average_precision.value
    )
    assert (
        body["metrics"]["averagePrecision"]["ciLower"]
        == known_result.metrics.average_precision.ci_lower
    )
    assert (
        body["metrics"]["averagePrecision"]["ciUpper"]
        == known_result.metrics.average_precision.ci_upper
    )
    assert body["confusionMatrix"]["tp"] == known_result.confusion_matrix.tp
    assert body["confusionMatrix"]["fp"] == known_result.confusion_matrix.fp
    assert body["confusionMatrix"]["tn"] == known_result.confusion_matrix.tn
    assert body["confusionMatrix"]["fn"] == known_result.confusion_matrix.fn


def test_evaluate_endpoint_returns_400_when_artifacts_missing(
    client: TestClient,
) -> None:
    container = configure_container()
    failing_service = MagicMock(spec=EvaluationService)
    failing_service.evaluate.side_effect = NoEvaluationArtifactsError

    with container.evaluation_service.override(failing_service):
        response = client.get("/evaluate/")

    assert response.status_code == status.HTTP_400_BAD_REQUEST
