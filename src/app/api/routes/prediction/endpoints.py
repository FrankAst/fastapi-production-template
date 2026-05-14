from typing import Annotated

from dependency_injector.wiring import inject
from fastapi import APIRouter, Body, File, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import PredictionServiceDependency
from app.domain import PredictionInput

from .examples import EXAMPLES
from .schemas import (
    BatchPredictionResponse,
    SinglePredictionRequest,
    SinglePredictionResponse,
    parse_prediction_upload,
)

router = APIRouter(prefix="/prediction", tags=["Prediction"])


@router.post("/single")
@inject
async def predict(
    prediction_request: Annotated[
        SinglePredictionRequest, Body(openapi_examples=EXAMPLES)
    ],
    prediction_service: PredictionServiceDependency,
) -> SinglePredictionResponse:
    """
    Score a single subject and return probability, bootstrap CI, screening
    flag, and the SHAP explanation in log-odds space.

    Args:
        prediction_request: Validated NHANES-shaped feature payload.
        prediction_service: Injected PredictionService.

    Returns:
        SinglePredictionResponse: Probability, CI bounds, screening flag,
        and the SHAP explanation.
    """
    validated_df = prediction_request.to_validated_dataframe()
    result = await run_in_threadpool(
        prediction_service.predict, PredictionInput(features=validated_df)
    )
    return SinglePredictionResponse.model_validate(result)


@router.post("/batch")
@inject
async def batch_predict(
    prediction_service: PredictionServiceDependency,
    file: Annotated[UploadFile, File(...)],
) -> BatchPredictionResponse:
    """
    Make batch predictions from an uploaded CSV file.

    Args:
        prediction_service: Injected service for handling predictions.
        file: CSV file containing feature data for batch prediction.

    Returns:
        BatchPredictionResponse: The prediction results and count.
    """
    feature_matrix = await parse_prediction_upload(file)

    prediction_input = PredictionInput(features=feature_matrix)
    prediction_output = await run_in_threadpool(
        prediction_service.batch_predict, prediction_input
    )

    return BatchPredictionResponse(
        predictions=prediction_output.predictions, count=prediction_output.count
    )
