import pytest

from app.services.evaluation import (
    EvaluationResult,
    EvaluationService,
    NoEvaluationArtifactsError,
)


def test_evaluate_returns_evaluation_result(
    evaluation_service: EvaluationService,
) -> None:
    result = evaluation_service.evaluate()

    assert isinstance(result, EvaluationResult)
    assert result.n_test > 0
    assert result.threshold == evaluation_service.lr_config.threshold


def test_evaluate_metrics_ci_brackets_are_valid(
    evaluation_service: EvaluationService,
) -> None:
    result = evaluation_service.evaluate()

    for metric in (
        result.metrics.average_precision,
        result.metrics.precision,
        result.metrics.recall,
        result.metrics.f1,
    ):
        assert metric.ci_lower <= metric.value <= metric.ci_upper
        assert metric.ci_lower >= 0.0
        assert metric.ci_upper <= 1.0


def test_evaluate_raises_when_artifacts_missing() -> None:
    with pytest.raises(NoEvaluationArtifactsError):
        EvaluationService().evaluate()
