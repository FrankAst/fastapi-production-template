from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.services import ArtifactPersistError


def artifact_persist_handler(
    _: Request,
    __: ArtifactPersistError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Failed to persist training artifact"},
    )
