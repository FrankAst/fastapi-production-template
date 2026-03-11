from dependency_injector import containers, providers

from app.services import PredictionService, ProcessingService, TrainingService


class Container(containers.DeclarativeContainer):
    processing_service = providers.Factory(ProcessingService)
    prediction_service = providers.Factory(PredictionService)
    training_service = providers.Factory(
        TrainingService, processing_service=processing_service
    )
