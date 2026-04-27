from typing import Literal

from pydantic import BaseModel, ConfigDict
from sklearn.linear_model import LogisticRegression


class LRVifBicConfig(BaseModel):
    """Frozen config for the LR VIF+BIC model.
    Single source of truth for hyperparameters."""

    model_config = ConfigDict(frozen=True)

    penalty: None = None
    class_weight: str = "balanced"
    max_iter: int = 1000
    solver: Literal[
        "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
    ] = "lbfgs"
    threshold: float = 0.467
    n_bootstrap_train: int = 200
    n_bootstrap_eval: int = 2000
    n_shap_background: int = 100
    random_state: int = 37
    test_size: float = 0.2

    def create_estimator(self) -> LogisticRegression:
        """Construct the sklearn estimator from config hyperparameters.

        Returns:
            Configured LogisticRegression estimator.
        """
        return LogisticRegression(
            penalty=self.penalty,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            solver=self.solver,
        )
