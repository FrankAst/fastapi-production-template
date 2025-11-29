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
    # Create request object and process file
    train_request = await FileTrainRequest.from_upload(file)

    # Train the model
    training_service.train(train_request)

    return TrainResponse(
        message=f"Model trained successfully with {train_request.shape[0]} samples"
    )
