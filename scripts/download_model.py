#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: huggingface_hub. Install requirements first.") from exc

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "configs" / "ner.yaml"


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp) or {}
    return payload if isinstance(payload, dict) else {}


def _resolve_local_dir(system_cfg: Dict[str, Any]) -> Path:
    configured = Path(str(system_cfg.get("local_model_dir", "Sharifsetup-ner"))).expanduser()
    if configured.is_absolute():
        return configured
    return (ROOT_DIR / configured).resolve()


def main() -> None:
    cfg = _load_config(CONFIG_PATH)
    system_cfg = cfg.get("system", {})
    if not isinstance(system_cfg, dict):
        system_cfg = {}

    model_name = str(cfg.get("model_name", "fastino/gliner2-multi-v1"))
    cache_dir = Path(str(cfg.get("cache_dir", ".cache/ner"))).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = (ROOT_DIR / cache_dir).resolve()

    local_model_dir = _resolve_local_dir(system_cfg)
    revision = system_cfg.get("revision")

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_model_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    print(f"Downloading model: {model_name}")
    print(f"Cache directory: {cache_dir}")
    print(f"Local model directory: {local_model_dir}")

    kwargs: Dict[str, Any] = {
        "repo_id": model_name,
        "cache_dir": str(cache_dir),
        "local_dir": str(local_model_dir),
        "local_dir_use_symlinks": False,
    }
    if revision:
        kwargs["revision"] = str(revision)

    snapshot_download(**kwargs)
    print("Model download complete.")


if __name__ == "__main__":
    main()
