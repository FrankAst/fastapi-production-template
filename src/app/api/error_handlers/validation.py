from fastapi import Request, status
from fastapi.responses import JSONResponse
from pandera.errors import SchemaErrors

from app.domain import ClinicalConsistencyError, ErrorResponse


def schema_validation_handler(_: Request, exc: SchemaErrors) -> JSONResponse:
    failure_cases = (
        exc.failure_cases.to_dict("records") if hasattr(exc, "failure_cases") else None
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            detail="Data validation failed",
            failure_cases=failure_cases,
        ).model_dump(by_alias=True, exclude_none=True),
    )


def clinical_consistency_handler(
    _: Request, exc: ClinicalConsistencyError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            detail=str(exc),
            failure_cases=exc.failure_cases,
        ).model_dump(by_alias=True, exclude_none=True),
    )
