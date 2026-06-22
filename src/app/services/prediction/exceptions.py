from app.domain import AppError


class NoTrainedModelError(AppError):
    message: str = (
        "No trained model found. Please train the model before making predictions."
    )


class NoBootstrapEnsembleError(AppError):
    message: str = (
        "No bootstrap ensemble found. Please train the model before making predictions."
    )


class NoShapBackgroundError(AppError):
    message: str = (
        "No SHAP background dataset found. "
        "Please train the model before making predictions."
    )
