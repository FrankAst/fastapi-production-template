from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.services.evaluation import NoEvaluationArtifactsError


def no_evaluation_artifacts_handler(
    _: Request,
    exc: NoEvaluationArtifactsError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )
