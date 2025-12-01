from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.domain import MLModel, PredictionInput, PredictionOutput
from app.services.helper import load_model
from app.services.processing.service import ProcessingService
from app.settings import Settings

from .exceptions import NoTrainedModelError


class PredictionService(BaseModel):
    model_path: Path = Field(default=Settings.MODEL_PATH)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def model(self) -> MLModel | None:
        return load_model(self.model_path)

    def predict(self, prediction_input: PredictionInput) -> PredictionOutput:
        """
        Make predictions for a single or batch input.

        Args:
            prediction_input: PredictionInput instance containing
            features for prediction.

        Returns:
            PredictionOutput: Contains predictions and count.

        Raises:
            NoTrainedModelError: If no trained model is found.
        """
        if self.model is None:
            raise NoTrainedModelError
        # Preprocess features
        processed_features = ProcessingService.preprocess(
            prediction_input.features, skip=False
        )

        # Make predictions
        prediction_results = self.model.predict(processed_features)

        return PredictionOutput(
            predictions=prediction_results, count=len(prediction_results)
        )
