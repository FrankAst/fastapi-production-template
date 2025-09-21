from typing import Annotated

from dependency_injector.wiring import inject
from fastapi import APIRouter, Body, File, UploadFile

from app.api.dependencies import PredictionServiceDependency
from app.domain import PredictionInput

from .examples import EXAMPLES
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    SinglePredictionRequest,
    SinglePredictionResponse,
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
    Make a single prediction from an array of features.

    Args:
        prediction_request: The input features for making a prediction.
        prediction_service: Injected service for handling predictions.

    Returns:
        SinglePredictionResponse: The prediction result for the input features.
    """
    feature_matrix = PredictionInput(features=[prediction_request.features])
    prediction = prediction_service.predict(feature_matrix)
    return SinglePredictionResponse(prediction=prediction.predictions[0])


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
    batch_request = await BatchPredictionRequest.from_upload(file)
    feature_matrix = await batch_request.to_feature_matrix()

    # Create PredictionInput and perform predictions
    prediction_input = PredictionInput(features=feature_matrix)
    prediction_output = prediction_service.predict(prediction_input)

    return BatchPredictionResponse(
        predictions=prediction_output.predictions, count=prediction_output.count
    )
