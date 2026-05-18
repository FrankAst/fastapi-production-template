from collections.abc import Iterable

from fastapi import APIRouter

from .health import health_router
from .prediction import (
    SinglePredictionRequest,
    SinglePredictionResponse,
    prediction_router,
)
from .train import TrainResponse, train_router

ROUTERS: Iterable[APIRouter] = (
    health_router,
    prediction_router,
    train_router,
)

__all__ = [
    "ROUTERS",
    "SinglePredictionRequest",
    "SinglePredictionResponse",
    "TrainResponse",
    "health_router",
    "prediction_router",
    "train_router",
]
