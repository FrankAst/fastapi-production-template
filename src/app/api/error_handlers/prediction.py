from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.domain import ErrorResponse
from app.services.prediction import (
    NoBootstrapEnsembleError,
    NoShapBackgroundError,
    NoTrainedModelError,
)


def no_trained_model_handler(_: Request, exc: NoTrainedModelError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content=ErrorResponse(detail=str(exc)).model_dump(
            by_alias=True, exclude_none=True
        ),
    )


def no_bootstrap_ensemble_handler(
    _: Request, exc: NoBootstrapEnsembleError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content=ErrorResponse(detail=str(exc)).model_dump(
            by_alias=True, exclude_none=True
        ),
    )


def no_shap_background_handler(_: Request, exc: NoShapBackgroundError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content=ErrorResponse(detail=str(exc)).model_dump(
            by_alias=True, exclude_none=True
        ),
    )
