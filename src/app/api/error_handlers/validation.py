"""Error handlers for data validation exceptions."""

import json

from fastapi import Request
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
    # Try to parse the error message as JSON for better readability
    try:
        error_details = json.loads(str(exc))
    except json.JSONDecodeError:
        # If not valid JSON, use the string as-is
        error_details = str(exc)

    return JSONResponse(
        status_code=422,
        content={
            "error": "Data validation failed",
            "details": error_details,
            "failure_cases": exc.failure_cases.to_dict("records")  # type: ignore  # noqa: PGH003
            if hasattr(exc, "failure_cases")
            else None,
        },
    )
