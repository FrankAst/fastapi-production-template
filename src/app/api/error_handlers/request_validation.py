from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.domain import ErrorResponse


def request_validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            detail="Request body validation failed",
            failure_cases=list(exc.errors()),
        ).model_dump(by_alias=True, exclude_none=True),
    )
