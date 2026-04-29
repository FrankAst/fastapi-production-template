"""Error handlers for data validation exceptions."""

import json

from fastapi import Request, status
from fastapi.responses import JSONResponse
from pandera.errors import SchemaErrors


def schema_validation_handler(
    _: Request,
    exc: SchemaErrors,
) -> JSONResponse:
    """
    Handle Pandera schema validation errors with detailed failure information.

    Returns:
        JSONResponse: 422 response with validation error details.
    """
    try:
        error_details = json.loads(str(exc))
    except json.JSONDecodeError:
        error_details = str(exc)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Data validation failed",
            "details": error_details,
            "failure_cases": exc.failure_cases.to_dict("records")
            if hasattr(exc, "failure_cases")
            else None,
        },
    )
