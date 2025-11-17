from __future__ import annotations

from collections.abc import Callable

from fastapi import Request, Response
from pandera.errors import SchemaErrors  # pylint: disable=import-error

from app.domain import NoTrainingSchemaError
from app.services import NoTrainedModelError
from app.services.training import DimensionalityMismatchError

from .domain import no_trained_model_handler, no_training_schema_handler
from .training import dimensionality_mismatch_handler
from .validation import schema_validation_handler

ExceptionHandler = Callable[[Request, Exception], Response]

EXCEPTION_HANDLERS: dict[type[Exception], ExceptionHandler] = {
    DimensionalityMismatchError: dimensionality_mismatch_handler,  # type: ignore[dict-item]
    NoTrainedModelError: no_trained_model_handler,  # type: ignore[dict-item]
    NoTrainingSchemaError: no_training_schema_handler,  # type: ignore[dict-item]
    SchemaErrors: schema_validation_handler,
}

__all__ = ["EXCEPTION_HANDLERS"]
