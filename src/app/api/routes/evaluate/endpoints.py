from dependency_injector.wiring import inject
from fastapi import APIRouter

from app.api.dependencies import EvaluationServiceDependency

from .schemas import (
    ConfusionMatrixSchema,
    EvaluateResponse,
    MetricsSchema,
    MetricWithCISchema,
)

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])


@router.get("/")
@inject
async def evaluate(evaluation_service: EvaluationServiceDependency) -> EvaluateResponse:
    result = evaluation_service.evaluate()

    return EvaluateResponse(
        threshold=result.threshold,
        n_test=result.n_test,
        metrics=MetricsSchema(
            average_precision=MetricWithCISchema(
                value=result.metrics.average_precision.value,
                ci_lower=result.metrics.average_precision.ci_lower,
                ci_upper=result.metrics.average_precision.ci_upper,
            ),
            precision=MetricWithCISchema(
                value=result.metrics.precision.value,
                ci_lower=result.metrics.precision.ci_lower,
                ci_upper=result.metrics.precision.ci_upper,
            ),
            recall=MetricWithCISchema(
                value=result.metrics.recall.value,
                ci_lower=result.metrics.recall.ci_lower,
                ci_upper=result.metrics.recall.ci_upper,
            ),
            f1=MetricWithCISchema(
                value=result.metrics.f1.value,
                ci_lower=result.metrics.f1.ci_lower,
                ci_upper=result.metrics.f1.ci_upper,
            ),
        ),
        confusion_matrix=ConfusionMatrixSchema(
            tp=result.confusion_matrix.tp,
            fp=result.confusion_matrix.fp,
            tn=result.confusion_matrix.tn,
            fn=result.confusion_matrix.fn,
        ),
    )
