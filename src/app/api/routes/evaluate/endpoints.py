from dependency_injector.wiring import inject
from fastapi import APIRouter

from app.api.dependencies import EvaluationServiceDependency

from .schemas import EvaluateResponse

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])


@router.get("/")
@inject
async def evaluate(evaluation_service: EvaluationServiceDependency) -> EvaluateResponse:
    return EvaluateResponse.model_validate(evaluation_service.evaluate())
