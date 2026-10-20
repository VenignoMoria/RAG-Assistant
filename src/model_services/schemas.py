from typing import Literal
from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32)
    kind: Literal["query", "passage"]


class RerankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    texts: list[str] = Field(min_length=1, max_length=20)


class NLIRequest(BaseModel):
    premise: str = Field(min_length=1, max_length=10000)
    hypothesis: str = Field(min_length=1, max_length=1500)
