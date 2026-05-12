from dependency_injector.wiring import inject
from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import EvaluationServiceDependency

from .responses import RESPONSES
from .schemas import EvaluateResponse

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])


@router.get("/", responses=RESPONSES)
@inject
async def evaluate(evaluation_service: EvaluationServiceDependency) -> EvaluateResponse:
    result = await run_in_threadpool(evaluation_service.evaluate)
    return EvaluateResponse.model_validate(result)
