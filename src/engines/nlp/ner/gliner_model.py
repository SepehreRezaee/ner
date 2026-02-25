from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .interfaces import IEModel, NerConfig, NerEntity

logger = logging.getLogger(__name__)


def _collect_exception_messages(exc: BaseException) -> str:
    parts: list[str] = []
    seen: set[int] = set()
    current: Optional[BaseException] = exc

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).strip()
        if message:
            parts.append(message.lower())
        next_exc = current.__cause__ or current.__context__
        current = next_exc if isinstance(next_exc, BaseException) else None

    return " | ".join(parts)


def _is_numpy_abi_mismatch(exc: BaseException) -> bool:
    message_chain = _collect_exception_messages(exc)
    markers = (
        "compiled using numpy 1.x",
        "cannot be run in numpy 2",
        "numpy.core.multiarray failed to import",
        "multiarray failed to import",
    )
    return any(marker in message_chain for marker in markers)


def _ensure_transformers_sklearn_compat() -> None:
    """Disable sklearn-dependent transformers paths when sklearn import is broken."""
    try:
        import_utils = importlib.import_module("transformers.utils.import_utils")
    except Exception:
        return

    try:
        importlib.import_module("sklearn")
    except ModuleNotFoundError:
        return
    except Exception as exc:
        setattr(import_utils, "_sklearn_available", False)
        logger.warning(
            "Detected unusable sklearn installation (%s). "
            "Disabling sklearn-dependent transformers features.",
            exc.__class__.__name__,
        )


def _ensure_transformers_compat() -> None:
    """Patch known lazy-import regressions and optional dependency issues."""
    _ensure_transformers_sklearn_compat()
    try:
        integrations = importlib.import_module("transformers.integrations")
    except Exception:
        return

    try:
        getattr(integrations, "get_reporting_integration_callbacks")
        return
    except Exception:
        pass

    callback_fn: Any
    try:
        integration_utils = importlib.import_module("transformers.integrations.integration_utils")
        callback_fn = getattr(integration_utils, "get_reporting_integration_callbacks")
    except Exception:
        logger.warning(
            "transformers.integrations.get_reporting_integration_callbacks is unavailable; "
            "using a no-op compatibility fallback."
        )

        def callback_fn(*_args: Any, **_kwargs: Any) -> list[Any]:
            return []

    setattr(integrations, "get_reporting_integration_callbacks", callback_fn)


def _import_gliner2() -> Any:
    _ensure_transformers_compat()
    try:
        module = importlib.import_module("gliner2")
        return getattr(module, "GLiNER2")
    except Exception as exc:
        if _is_numpy_abi_mismatch(exc):
            raise RuntimeError(
                "Failed to import gliner2 due to an incompatible NumPy/SciPy/sklearn binary stack "
                "(NumPy 2.x with extensions built for NumPy 1.x). "
                "Install `numpy<2` and reinstall dependencies from requirements.txt."
            ) from exc
        raise RuntimeError(
            "Failed to import gliner2. This usually means your gliner/gliner2/transformers "
            "versions are incompatible. Reinstall dependencies from requirements.txt and retry."
        ) from exc


class GlinerModel(IEModel):
    def __init__(self, config: NerConfig):
        self.config = config
        self._model: Optional[object] = None
        self._did_warmup = False
        self._model_lock = asyncio.Lock()
        self._warmup_lock = asyncio.Lock()
        self._bootstrap_lock = asyncio.Lock()
        self._local_model_dir = self._resolve_local_model_dir()
        self._load_stats: Dict[str, Any] = {
            "status": "cold",
            "model_source": None,
            "device": None,
            "download_ms": None,
            "load_ms": None,
            "warmup_ms": None,
            "local_model_dir": str(self._local_model_dir),
            "loaded_from_local_snapshot": False,
            "error": None,
        }

    def _system_config(self) -> Dict[str, Any]:
        return self.config.system or {}

    @staticmethod
    def _to_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return default

    def _resolve_local_model_dir(self) -> Path:
        # Application root directory (project root in local dev, /app in Docker image).
        app_root = Path(__file__).resolve().parents[4]

        configured_dir = self._system_config().get("local_model_dir", "Sharifsetup-ner")
        local_dir = Path(configured_dir).expanduser()
        if not local_dir.is_absolute():
            local_dir = (app_root / local_dir).resolve()
        return local_dir

    def _has_local_snapshot(self) -> bool:
        return self._local_model_dir.exists() and any(self._local_model_dir.iterdir())

    def _resolve_offline_mode(self, has_snapshot: bool) -> bool:
        system = self._system_config()
        offline_cfg = system.get("offline_mode", "auto")
        env_offline = os.getenv("HF_HUB_OFFLINE", "0") == "1"

        if isinstance(offline_cfg, bool):
            return offline_cfg
        if isinstance(offline_cfg, str):
            option = offline_cfg.strip().lower()
            if option in {"1", "true", "yes", "on"}:
                return True
            if option in {"0", "false", "no", "off"}:
                return False
        prefer_local = self._to_bool(system.get("prefer_local_files", True), default=True)
        return env_offline or (prefer_local and has_snapshot)

    def _apply_hf_environment(self, offline_mode: bool) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if offline_mode:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

    @staticmethod
    def _filter_supported_kwargs(func: Any, kwargs: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return kwargs, []

        accepts_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        )
        if accepts_var_kwargs:
            return kwargs, []

        allowed = {name for name in signature.parameters.keys() if name != "self"}
        filtered: Dict[str, Any] = {}
        dropped: List[str] = []
        for key, value in kwargs.items():
            if key in allowed:
                filtered[key] = value
            else:
                dropped.append(key)
        return filtered, dropped

    def _resolve_device(self) -> str:
        import torch

        requested = str(self._system_config().get("device", "auto")).strip().lower()
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if has_mps:
                return "mps"
            return "cpu"

        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("Config requested CUDA but it is not available. Falling back to CPU.")
            return "cpu"
        if requested == "mps":
            has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if not has_mps:
                logger.warning("Config requested MPS but it is not available. Falling back to CPU.")
                return "cpu"
        return requested

    def _resolve_torch_dtype(self, device: str) -> Any:
        import torch

        dtype_cfg = self._system_config().get("torch_dtype", "auto")
        if dtype_cfg is None:
            return None

        dtype_key = str(dtype_cfg).strip().lower()
        if dtype_key in {"", "none"}:
            return None
        if dtype_key == "auto":
            if device.startswith("cuda") or device.startswith("mps"):
                return torch.float16
            return torch.float32

        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        resolved = mapping.get(dtype_key)
        if resolved is None:
            logger.warning("Unknown torch_dtype '%s'; ignoring.", dtype_cfg)
        return resolved

    def get_runtime_stats(self) -> Dict[str, Any]:
        snapshot = dict(self._load_stats)
        snapshot["has_local_snapshot"] = self._has_local_snapshot()
        snapshot["model_loaded"] = self._model is not None
        snapshot["did_warmup"] = self._did_warmup
        return snapshot

    def _download_snapshot_sync(self) -> Path:
        start = time.perf_counter()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self._local_model_dir.mkdir(parents=True, exist_ok=True)

        if self._has_local_snapshot():
            logger.info("Using existing local model snapshot: %s", self._local_model_dir)
            return self._local_model_dir

        if self._resolve_offline_mode(has_snapshot=False):
            raise RuntimeError(
                f"Offline mode enabled, but local model snapshot was not found at {self._local_model_dir}."
            )

        from huggingface_hub import snapshot_download

        revision = self._system_config().get("revision")
        logger.info("Downloading model snapshot '%s' to %s", self.config.model_name, self._local_model_dir)
        download_kwargs: Dict[str, Any] = {
            "repo_id": self.config.model_name,
            "cache_dir": str(self.config.cache_dir),
            "local_dir": str(self._local_model_dir),
            "local_dir_use_symlinks": False,
        }
        if revision:
            download_kwargs["revision"] = revision
        snapshot_download(**download_kwargs)
        self._load_stats["download_ms"] = (time.perf_counter() - start) * 1000
        return self._local_model_dir

    async def prepare_local_model(self, download: bool = True) -> Path:
        async with self._bootstrap_lock:
            if self._has_local_snapshot():
                return self._local_model_dir

            if not download:
                raise RuntimeError(
                    f"Local model snapshot not found at {self._local_model_dir}. "
                    "Enable startup download or download it manually."
                )

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._download_snapshot_sync)

    def _resolve_model_source(self) -> str:
        self._load_stats["loaded_from_local_snapshot"] = self._has_local_snapshot()
        # Always load from the explicit local model directory.
        return str(self._local_model_dir)

    async def load(self) -> None:
        if self._model is not None:
            return

        async with self._model_lock:
            if self._model is not None:
                return

            if not self._has_local_snapshot():
                download_allowed = self._to_bool(
                    self._system_config().get("download_on_startup", True),
                    default=True,
                )
                if not download_allowed:
                    raise RuntimeError(
                        f"Local model snapshot not found at {self._local_model_dir}. "
                        "Set system.download_on_startup=true to auto-download it."
                    )
                await self.prepare_local_model(download=True)

            load_start = time.perf_counter()
            source = self._resolve_model_source()
            self._load_stats["status"] = "loading"
            self._load_stats["model_source"] = source
            self._load_stats["error"] = None
            logger.info("Loading GLiNER2 model from: %s", source)
            loop = asyncio.get_running_loop()
            try:
                self._model = await loop.run_in_executor(None, self._load_sync)
            except Exception as exc:
                self._load_stats["status"] = "failed"
                self._load_stats["error"] = str(exc)
                raise

            self._load_stats["load_ms"] = (time.perf_counter() - load_start) * 1000
            self._load_stats["status"] = "loaded"
            logger.info(
                "GLiNER2 model loaded in %.2fms.",
                self._load_stats["load_ms"],
            )

        if self._to_bool(self._system_config().get("warmup_on_load", True), default=True):
            await self.warmup()
        else:
            self._load_stats["status"] = "ready"

    async def warmup(self) -> None:
        if self._did_warmup:
            return
        if self._model is None:
            await self.load()
            return

        async with self._warmup_lock:
            if self._did_warmup:
                return
            self._load_stats["status"] = "warming"
            warmup_start = time.perf_counter()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._warmup_sync)
            self._load_stats["warmup_ms"] = (time.perf_counter() - warmup_start) * 1000
            self._did_warmup = True
            self._load_stats["status"] = "ready"
            logger.info("GLiNER2 warmup finished in %.2fms.", self._load_stats["warmup_ms"])

    def _warmup_sync(self) -> None:
        if self._model is None:
            return

        system = self._system_config()
        warmup_text = str(
            system.get(
                "warmup_text",
                "On February 17, 2026, Sharifsetup-NER launched NER Studio in San Francisco.",
            )
        ).strip()
        warmup_labels = system.get(
            "warmup_labels",
            ["person", "organization", "location", "date", "time", "money"],
        )
        if not warmup_text or not warmup_labels:
            return

        try:
            self._model.extract_entities(
                warmup_text,
                warmup_labels,
                include_confidence=False,
                include_spans=False,
            )
        except TypeError:
            self._model.extract_entities(warmup_text, warmup_labels)
        except Exception as exc:
            logger.warning("Warmup inference failed, continuing without warmup: %s", exc)

    def _load_sync(self) -> object:
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        gliner2_cls = _import_gliner2()

        system_config = self._system_config()
        display_name = system_config.get("display_name", "Sharifsetup-NER")
        source = self._resolve_model_source()
        has_snapshot = self._has_local_snapshot()
        offline_mode = self._resolve_offline_mode(has_snapshot=has_snapshot)
        self._apply_hf_environment(offline_mode=offline_mode)

        import torch

        if "num_threads" in system_config:
            torch.set_num_threads(int(system_config["num_threads"]))
            logger.info("Set torch.num_threads to %d", int(system_config["num_threads"]))

        device = self._resolve_device()
        self._load_stats["device"] = device

        # Model source is always a local directory; keep transformers in local-only mode.
        local_files_only = True

        from_pretrained_kwargs: Dict[str, Any] = {
            "cache_dir": str(self.config.cache_dir),
            "local_files_only": local_files_only,
        }

        low_mem = self._to_bool(system_config.get("low_cpu_mem_usage", True), default=True)
        if low_mem:
            from_pretrained_kwargs["low_cpu_mem_usage"] = True

        revision = system_config.get("revision")
        if revision:
            from_pretrained_kwargs["revision"] = revision

        load_tokenizer = system_config.get("load_tokenizer")
        if load_tokenizer is not None:
            from_pretrained_kwargs["load_tokenizer"] = self._to_bool(load_tokenizer, default=True)

        trust_remote_code = system_config.get("trust_remote_code")
        if trust_remote_code is not None:
            from_pretrained_kwargs["trust_remote_code"] = self._to_bool(trust_remote_code, default=False)

        device_map = system_config.get("device_map")
        if device_map not in (None, "", "none"):
            from_pretrained_kwargs["device_map"] = device_map

        torch_dtype = self._resolve_torch_dtype(device=device)
        if torch_dtype is not None:
            from_pretrained_kwargs["torch_dtype"] = torch_dtype

        filtered_kwargs, dropped = self._filter_supported_kwargs(gliner2_cls.from_pretrained, from_pretrained_kwargs)
        if dropped:
            logger.debug("Skipping unsupported GLiNER2.from_pretrained kwargs: %s", dropped)

        logger.info(
            "Loading NER model '%s' from %s (device=%s, local_files_only=%s, low_cpu_mem_usage=%s)",
            display_name,
            source,
            device,
            local_files_only,
            low_mem,
        )

        try:
            model = gliner2_cls.from_pretrained(source, **filtered_kwargs)
        except TypeError as exc:
            logger.warning(
                "GLiNER2.from_pretrained rejected kwargs (%s). Retrying with minimal kwargs.",
                exc,
            )
            minimal_kwargs = {
                key: value
                for key, value in filtered_kwargs.items()
                if key in {"local_files_only", "revision"}
            }
            model = gliner2_cls.from_pretrained(source, **minimal_kwargs)

        model.to(device)
        logger.info("NER model '%s' loaded and moved to %s.", display_name, device)
        return model

    async def predict(self, text: str, labels: List[str] | Dict[str, str]) -> List[NerEntity]:
        if self._model is None:
            await self.load()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.predict_sync(text, labels))

    async def predict_batch(self, texts: List[str], labels: List[str] | Dict[str, str]) -> List[List[NerEntity]]:
        if self._model is None:
            await self.load()

        if not texts:
            return []

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.predict_batch_sync(texts, labels))

    def predict_batch_sync(self, texts: List[str], labels: List[str] | Dict[str, str]) -> List[List[NerEntity]]:
        if self._model is None:
            raise RuntimeError("Model not loaded.")

        # Preserve label definitions when provided; fallback to label names if backend rejects dict labels.
        processed_labels: List[str] | Dict[str, str] = labels

        try:
            if hasattr(self._model, "batch_extract_entities"):
                all_raw_results = self._model.batch_extract_entities(
                    texts,
                    processed_labels,
                    include_confidence=True,
                    include_spans=True,
                )
            else:
                all_raw_results = [
                    self._model.extract_entities(t, processed_labels, include_confidence=True, include_spans=True)
                    for t in texts
                ]

            return [self._convert_results(res) for res in all_raw_results]
        except Exception as exc:
            if isinstance(processed_labels, dict):
                logger.warning(
                    "Batch prediction with label definitions failed; retrying with label names only. Error: %s",
                    exc,
                )
                processed_labels = list(processed_labels.keys())
            else:
                logger.error("Error in batch prediction: %s", exc)

            results = []
            for t in texts:
                raw = self._model.extract_entities(t, processed_labels, include_confidence=True, include_spans=True)
                results.append(self._convert_results(raw))
            return results

    def predict_sync(self, text: str, labels: List[str] | Dict[str, str]) -> List[NerEntity]:
        if self._model is None:
            raise RuntimeError("Model not loaded.")

        if not text.strip():
            return []

        raw_results = self._model.extract_entities(
            text,
            labels,
            include_confidence=True,
            include_spans=True,
        )
        return self._convert_results(raw_results)

    async def classify(self, text: str, labels: Dict[str, List[str]], multi_label: bool = False) -> Dict[str, Any]:
        if self._model is None:
            await self.load()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._model.classify_text(
                text,
                labels,
                multi_label=multi_label,
                include_confidence=True,
            ),
        )

    async def extract_json(self, text: str, schema: Dict[str, List[str]]) -> Dict[str, Any]:
        if self._model is None:
            await self.load()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._model.extract_json(
                text,
                schema,
                include_confidence=True,
                include_spans=True,
            ),
        )

    async def extract_relations(self, text: str, labels: List[str]) -> Dict[str, Any]:
        if self._model is None:
            await self.load()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._model.extract_relations(
                text,
                labels,
                include_confidence=True,
                include_spans=True,
            ),
        )

    def _convert_results(self, raw_result: Dict[str, Any]) -> List[NerEntity]:
        entities: List[NerEntity] = []
        raw_entities: Any = raw_result
        if isinstance(raw_result, dict):
            raw_entities = raw_result.get("entities", raw_result)

        if isinstance(raw_entities, dict):
            for label, items in raw_entities.items():
                for item in items:
                    self._append_entity(entities, item, str(label))
        elif isinstance(raw_entities, list):
            for item in raw_entities:
                label = str(item.get("label", ""))
                self._append_entity(entities, item, label)

        return entities

    def _append_entity(self, entities: List[NerEntity], item: Dict[str, Any], label: str) -> None:
        display_name = self._system_config().get("display_name", "Sharifsetup-NER")
        entities.append(
            NerEntity(
                text=str(item.get("text", "")),
                label=label,
                start=int(item.get("start", 0)),
                end=int(item.get("end", 0)),
                score=float(item.get("confidence", item.get("score", 0.0))),
                source=display_name,
            )
        )
