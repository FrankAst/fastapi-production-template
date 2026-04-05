from collections.abc import Iterable

from fastapi import APIRouter

from .evaluate import EvaluateResponse, evaluate_router
from .health import health_router
from .prediction import (
    SinglePredictionRequest,
    SinglePredictionResponse,
    prediction_router,
)
from .train import TrainResponse, train_router

ROUTERS: Iterable[APIRouter] = (
    evaluate_router,
    health_router,
    prediction_router,
    train_router,
)

__all__ = [
    "ROUTERS",
    "EvaluateResponse",
    "SinglePredictionRequest",
    "SinglePredictionResponse",
    "TrainResponse",
    "evaluate_router",
    "health_router",
    "prediction_router",
    "train_router",
]
