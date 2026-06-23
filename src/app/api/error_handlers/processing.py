from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.services.processing import MissingFeatureColumnsError


def missing_feature_columns_handler(
    _: Request,
    exc: MissingFeatureColumnsError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )
