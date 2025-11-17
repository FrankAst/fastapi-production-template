"""Error handlers for domain layer exceptions."""

from fastapi import Request
from fastapi.responses import JSONResponse

from app.domain import NoTrainingSchemaError
from app.services import NoTrainedModelError


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
        status_code=400,
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
        status_code=400,
        content={"detail": exc.message},
    )
