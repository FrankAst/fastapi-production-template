from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sklearn.linear_model import LogisticRegression


class LRVifBicConfig(BaseModel):
    """Frozen config for the LR VIF+BIC model.

    Single source of truth for hyperparameters.

    These defaults are production-validated values from the offline experimentation
    phase (logged in W&B). The class is frozen by design: parameters change via PR
    with backing experimental evidence, not via API request.

    Notable defaults:

    - threshold = 0.467: from precision-recall analysis; lands recall close to 80%.
    - n_bootstrap_pred = 200: prediction-ensemble size for bootstrap CIs at inference.
      Offline convergence shows 95% PI bounds stabilise around B≈100-150.
    - n_bootstrap_eval = 1000: resample count for evaluation-metric CIs (Raschka
      Method 3). Offline convergence shows the 95% bounds stabilise around B≈1000.
    """

    model_config = ConfigDict(frozen=True)

    penalty: Literal["l1", "l2", "elasticnet"] | None = None
    class_weight: str = "balanced"
    max_iter: int = 1000
    solver: Literal[
        "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
    ] = "lbfgs"
    threshold: float = Field(default=0.467, gt=0.0, lt=1.0)
    n_bootstrap_pred: int = Field(default=200, gt=0)
    n_bootstrap_eval: int = Field(default=1000, gt=0)
    random_state: int = Field(default=37, gt=0)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)

    def create_estimator(self) -> LogisticRegression:
        """Construct the sklearn estimator from config hyperparameters.

        Returns:
            Configured LogisticRegression estimator.
        """
        return LogisticRegression(
            penalty=self.penalty,
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=self.max_iter,
            solver=self.solver,
        )
