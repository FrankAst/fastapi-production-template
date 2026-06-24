from app.domain import AppError


class MissingFeatureColumnsError(AppError):
    def __init__(self, missing_columns: list[str]) -> None:
        self.missing_columns = missing_columns
        super().__init__(f"Missing required feature columns: {missing_columns}")
