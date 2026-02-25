from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "ner.yaml"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _to_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp) or {}
    return payload if isinstance(payload, dict) else {}


def resolve_runtime_flags(config_path: Path | None = None) -> Dict[str, Any]:
    cfg_path = Path(os.getenv("NER_CONFIG_PATH", str(config_path or DEFAULT_CONFIG_PATH))).expanduser().resolve()
    payload = _load_yaml(cfg_path)
    system = payload.get("system", {})
    if not isinstance(system, dict):
        system = {}

    verbose_default = _to_bool(system.get("verbose_logs", False), default=False)
    env_verbose = os.getenv("VERBOSE_LOGS")
    verbose_logs = _to_bool(env_verbose, default=verbose_default) if env_verbose is not None else verbose_default

    return {
        "config_path": cfg_path,
        "verbose_logs": verbose_logs,
    }


def resolve_log_level(verbose_logs: bool) -> int:
    return logging.DEBUG if verbose_logs else logging.ERROR


def resolve_uvicorn_log_level(verbose_logs: bool) -> str:
    return "debug" if verbose_logs else "error"


def configure_logging(*, force: bool = False, config_path: Path | None = None) -> bool:
    flags = resolve_runtime_flags(config_path=config_path)
    verbose_logs = bool(flags["verbose_logs"])
    logging.basicConfig(
        level=resolve_log_level(verbose_logs),
        format=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_LOG_DATEFMT,
        force=force,
    )
    return verbose_logs

