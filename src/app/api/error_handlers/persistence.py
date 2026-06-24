from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.domain import ErrorResponse
from app.services import ArtifactPersistError


def artifact_persist_handler(_: Request, exc: ArtifactPersistError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(detail=str(exc)).model_dump(
            by_alias=True, exclude_none=True
        ),
    )
