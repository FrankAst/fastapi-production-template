import pytest

from app.domain import ShapContribution


def test_instantiates_with_feature_and_shap_value() -> None:
    feature = "waist_to_height_ratio"
    shap_value = -0.42

    contribution = ShapContribution(feature=feature, shap_value=shap_value)

    assert contribution.feature == feature
    assert contribution.shap_value == pytest.approx(shap_value)  # pyright: ignore[reportUnknownMemberType]
