from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Any, Dict
import yaml

@dataclass
class NerEntity:
    text: str
    label: str
    start: int
    end: int
    score: float
    source: str = "model"
    canonical_text: Optional[str] = None
    source_provenance: Optional[str] = None

@dataclass
class NerConfig:
    model_name: str
    cache_dir: Path
    
    # Nested configurations
    pipeline: Dict[str, Any]
    normalization: Dict[str, Any]
    post_canonicalization: Dict[str, Any]
    schema_passes: Dict[str, Any]
    enrichment: Dict[str, Any]
    conflict_resolution: Dict[str, Any]
    thresholds: Dict[str, Any]
    qa_checks: Dict[str, Any]
    domains: Dict[str, Dict[str, Any]] = None
    label_aliases: Dict[str, List[str]] = None
    system: Dict[str, Any] = None
    
    # Quick access fields (legacy support)
    split_lines: bool = True
    enable_normalization: bool = True
    confidence_threshold: float = 0.2

    @property
    def labels(self) -> Dict[str, str]:
        """Returns labels from the core_ner pass for backward compatibility."""
        return self.schema_passes.get("core_ner", {})

    @property
    def priorities(self) -> Dict[str, int]:
        """Returns priorities from conflict_resolution for backward compatibility."""
        return self.conflict_resolution.get("priorities", {})

    @classmethod
    def from_yaml(cls, path: Path) -> "NerConfig":
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            
        pipeline = data.get("pipeline", {})
        return cls(
            model_name=data.get("model_name", "fastino/gliner2-multi-v1"),
            cache_dir=Path(data.get("cache_dir", ".cache/ner")).expanduser(),
            pipeline=pipeline,
            normalization=data.get("normalization", {}),
            post_canonicalization=data.get("post_canonicalization", {}),
            schema_passes=data.get("schema_passes", {}),
            enrichment=data.get("enrichment", {}),
            conflict_resolution=data.get("conflict_resolution", {}),
            thresholds=data.get("thresholds", {}),
            qa_checks=data.get("qa_checks", {}),
            domains=data.get("domains", {}),
            label_aliases=data.get("label_aliases", {}),
            system=data.get("system", {}),
            split_lines=bool(pipeline.get("split_lines", True)),
            enable_normalization=bool(pipeline.get("enable_normalization", True)),
            confidence_threshold=float(pipeline.get("confidence_threshold", 0.2)),
        )

class IEModel(abc.ABC):
    """Abstract base class for Information Extraction models."""

    @abc.abstractmethod
    async def load(self) -> None:
        """Load the model resources."""
        pass

    @abc.abstractmethod
    async def predict(self, text: str, labels: List[str] | Dict[str, str]) -> List[NerEntity]:
        """Extract entities."""
        pass

    @abc.abstractmethod
    async def classify(self, text: str, labels: Dict[str, List[str]], multi_label: bool = False) -> Dict[str, Any]:
        """Classify text."""
        pass

    @abc.abstractmethod
    async def extract_json(self, text: str, schema: Dict[str, List[str]]) -> Dict[str, Any]:
        """Extract structured JSON data."""
        pass

    @abc.abstractmethod
    async def extract_relations(self, text: str, labels: List[str]) -> Dict[str, Any]:
        """Extract relations between entities."""
        pass

    @abc.abstractmethod
    async def predict_batch(self, texts: List[str], labels: List[str] | Dict[str, str]) -> List[List[NerEntity]]:
        """Extract entities from multiple texts in parallel/batch."""
        pass

    @abc.abstractmethod
    def predict_sync(self, text: str, labels: List[str] | Dict[str, str]) -> List[NerEntity]:
        """Synchronous NER prediction."""
        pass
