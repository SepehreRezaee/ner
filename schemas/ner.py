from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class NerEntitySchema(BaseModel):
    text: str
    label: str
    start: int
    end: int
    score: float = Field(ge=0, le=1)
    source: Optional[str] = None


class UsageDetails(BaseModel):
    char_count: int
    execution_time_ms: float


class NerResponse(BaseModel):
    language: str
    entities: List[NerEntitySchema]
    usage: UsageDetails


class TextRequest(BaseModel):
    text: str
    language: Optional[str] = None
    labels: Optional[List[str]] = None