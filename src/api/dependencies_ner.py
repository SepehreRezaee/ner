"""FastAPI dependency providers and runtime state for NER service."""

from __future__ import annotations

import asyncio
import logging
import time
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from src.engines.nlp.ner import NerService, build_service

logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "configs" / "ner.yaml"

_RUNTIME_STATE_LOCK = Lock()
_WARMUP_LOCKS: Dict[int, asyncio.Lock] = {}
_RUNTIME_STATE: Dict[str, Any] = {
    "status": "cold",
    "ready": False,
    "started_at_epoch": None,
    "finished_at_epoch": None,
    "duration_ms": None,
    "last_error": None,
}


@lru_cache(maxsize=1)
def get_ner_service() -> NerService:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"NER config not found at {CONFIG_PATH}")
    return build_service(CONFIG_PATH)


def _set_runtime_state(**updates: Any) -> None:
    with _RUNTIME_STATE_LOCK:
        _RUNTIME_STATE.update(updates)


def _snapshot_runtime_state() -> Dict[str, Any]:
    with _RUNTIME_STATE_LOCK:
        return dict(_RUNTIME_STATE)


def _get_warmup_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    with _RUNTIME_STATE_LOCK:
        lock = _WARMUP_LOCKS.get(loop_id)
        if lock is None:
            lock = asyncio.Lock()
            _WARMUP_LOCKS[loop_id] = lock
    return lock


def get_startup_mode() -> str:
    try:
        service = get_ner_service()
        mode = str((service.config.system or {}).get("startup_mode", "blocking")).strip().lower()
    except Exception:
        return "blocking"
    return mode if mode in {"blocking", "background"} else "blocking"


def get_ner_runtime_state() -> Dict[str, Any]:
    payload = _snapshot_runtime_state()
    try:
        service = get_ner_service()
        payload["service"] = service.get_runtime_status()
    except Exception as exc:
        payload["service"] = {}
        if not payload.get("last_error"):
            payload["last_error"] = str(exc)

    payload["ready"] = payload.get("status") == "ready"
    return payload


async def warmup_ner_service() -> NerService:
    async with _get_warmup_lock():
        current = _snapshot_runtime_state()
        if current.get("status") == "ready":
            return get_ner_service()

        service = get_ner_service()
        start_epoch = time.time()
        start_perf = time.perf_counter()

        _set_runtime_state(
            status="warming",
            ready=False,
            started_at_epoch=start_epoch,
            finished_at_epoch=None,
            duration_ms=None,
            last_error=None,
        )

        logger.info("Warming up NER service...")
        try:
            await service.startup()
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_perf) * 1000
            _set_runtime_state(
                status="failed",
                ready=False,
                finished_at_epoch=time.time(),
                duration_ms=duration_ms,
                last_error=str(exc),
            )
            logger.exception("NER service warmup failed")
            raise

        duration_ms = (time.perf_counter() - start_perf) * 1000
        _set_runtime_state(
            status="ready",
            ready=True,
            finished_at_epoch=time.time(),
            duration_ms=duration_ms,
            last_error=None,
        )
        logger.info("NER service warmup complete in %.2fms", duration_ms)
        return service


__all__ = [
    "get_ner_service",
    "get_ner_runtime_state",
    "get_startup_mode",
    "warmup_ner_service",
]
