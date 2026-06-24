from app.domain import AppError


class NoEvaluationArtifactsError(AppError):
    message: str = (
        "No evaluation artifacts found. "
        "Please train the model first to generate evaluation outputs."
    )
