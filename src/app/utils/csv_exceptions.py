from app.domain.exceptions import AppError


class CsvFormatError(AppError):
    """Raised when CSV upload cannot be read or parsed."""


class CsvSizeError(AppError):
    """Raised when the upload exceeds the configured size limit."""


class CsvContentError(AppError):
    """Raised when the CSV parses but contains invalid content."""
