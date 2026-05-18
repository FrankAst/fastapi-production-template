from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.services.training import DimensionalityMismatchError


def dimensionality_mismatch_handler(
    _: Request,
    exc: DimensionalityMismatchError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )
