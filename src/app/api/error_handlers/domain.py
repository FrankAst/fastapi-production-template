from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.domain import NoTrainingSchemaError


def no_training_schema_handler(_: Request, exc: NoTrainingSchemaError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )
