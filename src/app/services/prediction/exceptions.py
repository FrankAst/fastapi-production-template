from dataclasses import dataclass


@dataclass
class NoTrainedModelError(Exception):
    message: str = (
        "No trained model found. Please train the model before making predictions."
    )


@dataclass
class NoBootstrapEnsembleError(Exception):
    message: str = (
        "No bootstrap ensemble found. Please train the model before making predictions."
    )


@dataclass
class NoShapBackgroundError(Exception):
    message: str = (
        "No SHAP background dataset found. "
        "Please train the model before making predictions."
    )
