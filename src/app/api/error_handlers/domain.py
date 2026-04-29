"""Error handlers for domain layer exceptions."""

from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.domain import NoTrainingSchemaError
from app.services.prediction import (
    NoBootstrapEnsembleError,
    NoShapBackgroundError,
    NoTrainedModelError,
)


def no_training_schema_handler(
    _: Request,
    exc: NoTrainingSchemaError,
) -> JSONResponse:
    """
    Handle missing training schema errors.

    Returns:
        JSONResponse: 400 response indicating model needs to be trained.
    """
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


def no_trained_model_handler(
    _: Request,
    exc: NoTrainedModelError,
) -> JSONResponse:
    """
    Handle missing trained model errors.

    Returns:
        JSONResponse: 400 response indicating model needs to be trained.
    """
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": exc.message},
    )


def no_bootstrap_ensemble_handler(
    _: Request,
    exc: NoBootstrapEnsembleError,
) -> JSONResponse:
    """
    Handle missing bootstrap ensemble errors.

    Returns:
        JSONResponse: 400 response indicating model needs to be trained.
    """
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": exc.message},
    )


def no_shap_background_handler(
    _: Request,
    exc: NoShapBackgroundError,
) -> JSONResponse:
    """
    Handle missing SHAP background dataset errors.

    Returns:
        JSONResponse: 400 response indicating model needs to be trained.
    """
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": exc.message},
    )
