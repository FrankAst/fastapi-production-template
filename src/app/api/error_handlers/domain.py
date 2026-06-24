from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.domain import ErrorResponse, NoTrainingSchemaError


def no_training_schema_handler(_: Request, exc: NoTrainingSchemaError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content=ErrorResponse(detail=str(exc)).model_dump(
            by_alias=True, exclude_none=True
        ),
    )
