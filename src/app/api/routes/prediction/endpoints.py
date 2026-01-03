from typing import Annotated

from dependency_injector.wiring import inject
from fastapi import APIRouter, Body, File, HTTPException, UploadFile

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
async def predict(
    _prediction_request: Annotated[
        SinglePredictionRequest, Body(openapi_examples=EXAMPLES)
    ],
    _prediction_service: PredictionServiceDependency,
) -> SinglePredictionResponse:
    """
    Placeholder for single predictions from UI.

    This endpoint is reserved for future UI integration where users will input
    individual feature values through a web interface. The implementation will
    be completed once the UI requirements and input structure are defined.

    Raises:
        HTTPException: 501 Not Implemented - Use /batch endpoint for now.
    """
    raise HTTPException(
        status_code=501,
        detail="Single prediction endpoint not yet implemented. Use /batch for now.",
    )


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
    # Dataset loading and validation
    feature_matrix = await BatchPredictionRequest.from_upload(file)
    # In the future there might be additional processing steps here

    # Create PredictionInput and perform predictions
    prediction_input = PredictionInput(features=feature_matrix)
    prediction_output = prediction_service.predict(prediction_input)

    return BatchPredictionResponse(
        predictions=prediction_output.predictions, count=prediction_output.count
    )
