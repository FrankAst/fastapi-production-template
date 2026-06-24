from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.domain import ErrorResponse
from app.services.evaluation import NoEvaluationArtifactsError


def no_evaluation_artifacts_handler(
    _: Request, exc: NoEvaluationArtifactsError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content=ErrorResponse(detail=str(exc)).model_dump(
            by_alias=True, exclude_none=True
        ),
    )
