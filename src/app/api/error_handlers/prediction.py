from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.services.prediction import (
    NoBootstrapEnsembleError,
    NoShapBackgroundError,
    NoTrainedModelError,
)


def no_trained_model_handler(_: Request, exc: NoTrainedModelError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


def no_bootstrap_ensemble_handler(
    _: Request, exc: NoBootstrapEnsembleError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


def no_shap_background_handler(_: Request, exc: NoShapBackgroundError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )
