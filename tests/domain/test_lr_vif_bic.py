from sklearn.linear_model import LogisticRegression

from app.domain.models import LRVifBicConfig


def test_create_estimator_returns_configured_logistic_regression() -> None:
    config = LRVifBicConfig()
    estimator = config.create_estimator()

    params = estimator.get_params()

    assert isinstance(estimator, LogisticRegression)
    assert params["penalty"] == config.penalty
    assert params["class_weight"] == config.class_weight
    assert params["max_iter"] == config.max_iter
    assert params["solver"] == config.solver
