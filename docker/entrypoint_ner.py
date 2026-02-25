#!/usr/bin/env python3
"""Entrypoint for NER-only service."""

import logging
import os
import sys
from pathlib import Path

import torch

# Ensure project root and src are importable.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for p in (SRC_DIR, ROOT_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from src.core.logging_config import configure_logging, resolve_uvicorn_log_level

VERBOSE_LOGS_ENABLED = configure_logging(force=True)

logger = logging.getLogger(__name__)
logger.info("=== Starting NER Service Entrypoint ===")

# Patch torch.load to use weights_only=False by default (PyTorch 2.6 changed default to True)
logger.info("Patching torch.load for compatibility...")
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
logger.info("torch.load patched successfully")

# Also allowlist TorchVersion for weights_only loads (defensive).
try:
    from torch.torch_version import TorchVersion  # type: ignore
    torch.serialization.add_safe_globals([TorchVersion])  # type: ignore[attr-defined]
    logger.info("TorchVersion allowlisted")
except Exception as e:
    logger.warning("Could not allowlist TorchVersion: %s", e)

# Now start uvicorn
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn_log_level = resolve_uvicorn_log_level(VERBOSE_LOGS_ENABLED)

    logger.info("Starting uvicorn server on %s:%d", host, port)
    uvicorn.run(
        "api.app_ner:app",
        host=host,
        port=port,
        log_level=uvicorn_log_level,
        proxy_headers=True,
        forwarded_allow_ips="*",
        workers=1,
    )
