class AppError(Exception):
    """Base for errors translated to HTTP responses by error_handlers/."""

    message: str = "An application error occurred."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.message)


class FeaturesEmptyError(ValueError):
    def __init__(self) -> None:
        super().__init__("Features list must not be empty")


class FeaturesContainNaNError(ValueError):
    def __init__(self) -> None:
        super().__init__("Features list must not contain NaN values")


class NoTrainingSchemaError(AppError):
    message: str = (
        "No training schema found. Please train the model first to generate the schema."
    )


class ClinicalConsistencyError(AppError):
    """Raised when input violates clinical cross-field rules."""

    def __init__(self, failure_cases: list[dict[str, object]]) -> None:
        self.failure_cases = failure_cases
        super().__init__(
            "Input contains rows with physiologically inconsistent feature combinations"
        )
