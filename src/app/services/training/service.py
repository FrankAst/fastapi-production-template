from typing import cast

from pandas import DataFrame, Series
from pydantic import BaseModel, ConfigDict, Field
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

from app.domain import TARGET_COLUMN, EvaluationTestSet, LRVifBicConfig, MLModel
from app.services.helper import save_artifact, save_model
from app.services.processing import ProcessingService
from app.settings import Settings

from .training_result import TrainingResult


class TrainingService(BaseModel):
    processing_service: ProcessingService = Field(default_factory=ProcessingService)
    lr_config: LRVifBicConfig = Field(default_factory=LRVifBicConfig)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def train(self, df: DataFrame) -> TrainingResult:
        """
        Orchestrate the full training flow: evaluation, production,
        and bootstrap models.

        Args:
            df: Validated DataFrame containing features and target column.

        Returns:
            TrainingResult: Summary counts for the training run.
        """
        X, y = self._split_features_target(df)

        n_train, n_test = self._train_evaluation(X, y)
        self._train_production(X, y)
        self._train_bootstrap(X, y)

        return TrainingResult(
            n_samples=len(df),
            n_train=n_train,
            n_test=n_test,
            n_bootstrap=self.lr_config.n_bootstrap,
            threshold=self.lr_config.threshold,
        )

    @staticmethod
    def _split_features_target(df: DataFrame) -> tuple[DataFrame, Series]:
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        return X, y

    def _create_pipeline(self) -> Pipeline:
        return Pipeline([
            ("preprocessing", self.processing_service.pipeline),
            ("classifier", self.lr_config.create_estimator()),
        ])

    def _train_evaluation(self, X: DataFrame, y: Series) -> tuple[int, int]:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.lr_config.test_size,
            random_state=self.lr_config.random_state,
            stratify=y,
        )

        pipeline = self._create_pipeline()
        pipeline.fit(X_train, y_train)
        save_model(cast("MLModel", pipeline), Settings.EVAL_MODEL_PATH)
        save_artifact(
            EvaluationTestSet(X_test=X_test, y_test=y_test), Settings.TEST_SET_PATH
        )

        return len(X_train), len(X_test)

    def _train_production(self, X: DataFrame, y: Series) -> None:
        """Fit the production model on 100% of the data and persist artifacts.

        The post-processing schema is inferred here — from the production pipeline
        output — because this is the only model trained on the full dataset. The
        schema captures the exact feature distribution the prediction endpoint will
        validate against at inference time.
        """
        pipeline = self._create_pipeline()
        pipeline.fit(X, y)
        save_model(cast("MLModel", pipeline), Settings.PRODUCTION_MODEL_PATH)

    def _train_bootstrap(self, X: DataFrame, y: Series) -> None:
        ensemble = []
        for i in range(self.lr_config.n_bootstrap):
            X_resampled, y_resampled = resample(  # type: ignore[misc]
                X, y, replace=True, random_state=self.lr_config.random_state + i
            )
            X_resampled = cast("DataFrame", X_resampled).reset_index(drop=True)
            y_resampled = cast("Series", y_resampled).reset_index(drop=True)
            pipeline = self._create_pipeline()
            pipeline.fit(X_resampled, y_resampled)
            ensemble.append(pipeline)

        save_artifact(ensemble, Settings.BOOTSTRAP_ENSEMBLE_PATH)
