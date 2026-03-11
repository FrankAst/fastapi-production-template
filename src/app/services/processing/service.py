from .age_binner.transformer import AgeBinner
from .base import BasePreprocessor


# pylint: disable=too-few-public-methods
class ProcessingService:
    """Encapsulates preprocessing pipeline configuration."""

    @property
    def pipeline(self) -> BasePreprocessor:
        """Returns the preprocessing pipeline transformer.

        Returns:
            BasePreprocessor: Configured preprocessing transformer (AgeBinner).
        """
        return AgeBinner()
