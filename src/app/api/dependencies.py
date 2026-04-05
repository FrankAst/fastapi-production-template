from typing import Annotated

from dependency_injector.wiring import Provide
from fastapi import Depends

from app.services import EvaluationService, PredictionService, TrainingService

EvaluationServiceDependency = Annotated[
    EvaluationService,
    Depends(Provide["evaluation_service"]),
]

PredictionServiceDependency = Annotated[
    PredictionService,
    Depends(Provide["prediction_service"]),
]

TrainingServiceDependency = Annotated[
    TrainingService,
    Depends(Provide["training_service"]),
]
