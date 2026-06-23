from __future__ import annotations

from collections.abc import Callable

from fastapi import Request, Response
from pandera.errors import SchemaErrors

from app.domain import ClinicalConsistencyError, NoTrainingSchemaError
from app.services import ArtifactPersistError
from app.services.evaluation import NoEvaluationArtifactsError
from app.services.prediction import (
    NoBootstrapEnsembleError,
    NoShapBackgroundError,
    NoTrainedModelError,
)

from .domain import (
    no_bootstrap_ensemble_handler,
    no_shap_background_handler,
    no_trained_model_handler,
    no_training_schema_handler,
)
from .evaluation import no_evaluation_artifacts_handler
from .persistence import artifact_persist_handler
from .validation import clinical_consistency_handler, schema_validation_handler

ExceptionHandler = Callable[[Request, Exception], Response]

EXCEPTION_HANDLERS: dict[type[Exception], ExceptionHandler] = {
    ArtifactPersistError: artifact_persist_handler,  # type: ignore[dict-item]
    ClinicalConsistencyError: clinical_consistency_handler,  # type: ignore[dict-item]
    NoBootstrapEnsembleError: no_bootstrap_ensemble_handler,  # type: ignore[dict-item]
    NoEvaluationArtifactsError: no_evaluation_artifacts_handler,  # type: ignore[dict-item]
    NoShapBackgroundError: no_shap_background_handler,  # type: ignore[dict-item]
    NoTrainedModelError: no_trained_model_handler,  # type: ignore[dict-item]
    NoTrainingSchemaError: no_training_schema_handler,  # type: ignore[dict-item]
    SchemaErrors: schema_validation_handler,
}

__all__ = ["EXCEPTION_HANDLERS"]
