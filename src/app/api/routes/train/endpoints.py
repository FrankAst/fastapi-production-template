from typing import Annotated

from dependency_injector.wiring import inject
from fastapi import APIRouter, File, UploadFile

from app.api.dependencies import TrainingServiceDependency
from app.domain import LRVifBicConfig

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

    return TrainResponse(
        n_samples=result.n_samples,
        n_train=result.n_train,
        n_test=result.n_test,
        n_bootstrap=result.n_bootstrap,
        threshold=LRVifBicConfig().threshold,
    )
