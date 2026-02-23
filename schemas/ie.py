from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from .ner import UsageDetails, NerEntitySchema, NerResponse

# --- Classification ---

class ClassificationRequest(BaseModel):
    text: str
    labels: Dict[str, List[str]]
    language: Optional[str] = None
    multi_label: bool = False

class ClassificationResponse(BaseModel):
    language: str
    results: Dict[str, Any]
    usage: UsageDetails

# --- JSON Extraction ---

class JsonExtractionRequest(BaseModel):
    text: str
    schema_def: Dict[str, List[str]] = Field(..., alias="schema")
    language: Optional[str] = None

class JsonExtractionResponse(BaseModel):
    language: str
    data: Dict[str, Any]
    usage: UsageDetails

# --- Relation Extraction ---

class RelationExtractionRequest(BaseModel):
    text: str
    labels: List[str]
    language: Optional[str] = None

class RelationExtractionResponse(BaseModel):
    language: str
    relations: Dict[str, Any]
    usage: UsageDetails
