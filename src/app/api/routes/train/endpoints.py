from typing import Annotated

from dependency_injector.wiring import inject
from fastapi import APIRouter, File, UploadFile

from app.api.dependencies import TrainingServiceDependency

from .responses import RESPONSES
from .schemas import FileTrainRequest, TrainResponse

router = APIRouter(prefix="/train", tags=["Training"])


@router.post("/", responses=RESPONSES)
@inject
async def train(
    training_service: TrainingServiceDependency,
    file: Annotated[UploadFile, File(...)],
) -> TrainResponse:
    train_request = await FileTrainRequest.from_upload(file)
    result = training_service.train(train_request)
    return TrainResponse.model_validate(result)
