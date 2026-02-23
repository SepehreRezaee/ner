#!/usr/bin/env python3
"""Entrypoint for NER-only service."""

import logging
import sys
import torch

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)

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
    logger.info("Starting uvicorn server on 0.0.0.0:8000")
    logger.info("API will be available at http://0.0.0.0:8000")
    logger.info("API docs will be available at http://0.0.0.0:8000/docs")
    uvicorn.run("api.app_ner:app", host="0.0.0.0", port=8000, log_level="info")
