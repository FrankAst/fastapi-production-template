# Exceptions related to prediction input validation


class FeaturesEmptyError(ValueError):
    def __init__(self) -> None:
        super().__init__("Features list must not be empty")


class FeaturesContainNaNError(ValueError):
    def __init__(self) -> None:
        super().__init__("Features list must not contain NaN values")
