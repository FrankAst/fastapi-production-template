from typing import Annotated

from dependency_injector.wiring import inject
from fastapi import APIRouter, File, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import TrainingServiceDependency

from .responses import RESPONSES
from .schemas import TrainResponse, parse_training_upload

router = APIRouter(prefix="/train", tags=["Training"])


@router.post("/", responses=RESPONSES)
@inject
async def train(
    training_service: TrainingServiceDependency,
    file: Annotated[UploadFile, File(...)],
) -> TrainResponse:
    training_df = await parse_training_upload(file)
    result = await run_in_threadpool(training_service.train, training_df)
    return TrainResponse.model_validate(result)
