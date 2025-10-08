from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from app.utils import ExamplerMixIn


class BaseEntity(BaseModel, ExamplerMixIn):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        arbitrary_types_allowed=True,  # Allow types like pandas DataFrame
    )
