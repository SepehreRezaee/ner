"""Compatibility module for legacy imports.

The service implementation now lives in `service.py` to keep responsibilities clearer.
"""

from __future__ import annotations

from .interfaces import IEModel, NerConfig, NerEntity
from .service import IEService, NerResult, NerService, build_service

__all__ = [
    "IEModel",
    "NerEntity",
    "NerConfig",
    "NerResult",
    "IEService",
    "NerService",
    "build_service",
]
