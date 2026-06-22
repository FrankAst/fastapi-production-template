from pydantic import Field

from .base import BaseEntity


class ErrorResponse(BaseEntity):
    detail: str = Field(description="Human-readable error message.")
    failure_cases: list[dict[str, object]] | None = Field(
        default=None,
        description=(
            "Structured failure data; populated for validation errors that carry "
            "per-row or per-field context."
        ),
    )
