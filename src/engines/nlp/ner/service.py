"""High-performance NER/IE service core."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from src.core.usage import global_usage_tracker

from .gliner_model import GlinerModel
from .interfaces import IEModel, NerConfig, NerEntity
from .pipeline.enrichment import DeterministicEnricher
from .pipeline.qa import QASafetySuite
from .pipeline.resolver import ConflictResolver
from .utils.normalizer import Normalizer
from .utils.semantic_resolver import SemanticResolver

LabelDescriptions = Dict[str, str]
LabelSpec = Union[List[str], LabelDescriptions]


@dataclass
class NerResult:
    language: str
    entities: List[NerEntity]
    usage: Dict[str, Any]


class _IdentityOffsetMapper:
    @staticmethod
    def map_back(start: int, end: int) -> tuple[int, int]:
        return start, end


class IEService:
    """Single service entrypoint for NER + IE operations."""

    def __init__(self, config: NerConfig, model: IEModel) -> None:
        self.config = config
        self.model = model
        self._logger = logging.getLogger(__name__)

        self.enricher = DeterministicEnricher()
        self.resolver = ConflictResolver(
            priorities=config.priorities,
            config=config.enrichment.get("keyword_rules", {}),
        )
        self.normalizer = Normalizer(config.normalization)
        self.qa = QASafetySuite(config.qa_checks)

        pipeline_cfg = config.pipeline or {}
        perf_cfg = pipeline_cfg.get("performance", {})
        chunk_cfg = pipeline_cfg.get("chunking", {})

        self._batch_size = max(int(perf_cfg.get("batch_size", 8)), 1)
        self._request_concurrency_limit = max(int(pipeline_cfg.get("request_concurrency_limit", 8)), 1)
        self._inference_concurrency_limit = max(
            int(pipeline_cfg.get("inference_concurrency_limit", chunk_cfg.get("concurrency_limit", 4))),
            1,
        )
        default_batch_chars = int(chunk_cfg.get("max_chars", 850)) * self._batch_size
        self._batch_max_chars = max(int(perf_cfg.get("batch_max_chars", default_batch_chars)), 1)
        self._max_inflight_batches = max(
            int(perf_cfg.get("max_inflight_batches", self._inference_concurrency_limit * 2)),
            1,
        )

        self._request_semaphore = asyncio.Semaphore(self._request_concurrency_limit)
        self._inference_semaphore = asyncio.Semaphore(self._inference_concurrency_limit)

        self._domain_map = config.domains or {}
        self._label_alias_map = config.label_aliases or {}

        domain_registry = {key: value.get("aliases", []) for key, value in self._domain_map.items()}
        self.domain_resolver = SemanticResolver(domain_registry, threshold=0.8)
        self.label_resolver = SemanticResolver(self._label_alias_map, threshold=0.8)

        self._known_labels = self._build_known_label_set()
        self._default_label_groups = self._resolve_schema_label_groups("", include_gated=False)

        self._label_resolution_cache: Dict[tuple[tuple[str, ...], tuple[str, ...]], Optional[List[str]]] = {}

    def _build_known_label_set(self) -> set[str]:
        known: set[str] = set()

        for group in self._resolve_schema_label_groups("", include_gated=True):
            if isinstance(group, dict):
                known.update(group.keys())
            else:
                known.update(group)

        for domain_cfg in self._domain_map.values():
            for label in domain_cfg.get("labels", []):
                known.add(str(label).strip().lower())

        for canonical in self._label_alias_map.keys():
            known.add(str(canonical).strip().lower())

        return known

    async def startup(self) -> None:
        self._configure_rate_limit()

        download_on_startup = bool((self.config.system or {}).get("download_on_startup", True))
        prepare_local_model = getattr(self.model, "prepare_local_model", None)
        if callable(prepare_local_model):
            await prepare_local_model(download=download_on_startup)

        await self.model.load()

    def _configure_rate_limit(self) -> None:
        rate_cfg = (self.config.pipeline or {}).get("rate_limit", {})
        limit = int(rate_cfg.get("chars_per_window", global_usage_tracker.limit))
        window_seconds = int(rate_cfg.get("window_seconds", global_usage_tracker.window_seconds))
        global_usage_tracker.configure(limit=limit, window_seconds=window_seconds)

    def get_runtime_status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "usage_window_chars": global_usage_tracker.get_current_usage(),
            "usage_limit_chars": global_usage_tracker.limit,
            "usage_window_seconds": global_usage_tracker.window_seconds,
            "request_concurrency_limit": self._request_concurrency_limit,
            "inference_concurrency_limit": self._inference_concurrency_limit,
            "batch_size": self._batch_size,
            "batch_max_chars": self._batch_max_chars,
            "max_inflight_batches": self._max_inflight_batches,
        }
        get_model_stats = getattr(self.model, "get_runtime_stats", None)
        if callable(get_model_stats):
            status["model"] = get_model_stats()
        return status

    def _check_limit(self, text: str) -> None:
        usage = len(text)
        if not global_usage_tracker.add_usage(usage):
            raise RuntimeError(f"Rate limit exceeded. Current usage window full. Requested: {usage} chars.")

    async def process_text(
        self,
        text: str,
        *,
        language: Optional[str] = None,
        labels: Optional[List[str] | Dict[str, str]] = None,
        domains: Optional[List[str]] = None,
    ) -> NerResult:
        start_time = time.perf_counter()

        if not text:
            duration = (time.perf_counter() - start_time) * 1000
            return NerResult(
                language=language or "unknown",
                entities=[],
                usage={"char_count": 0, "execution_time_ms": duration},
            )

        self._check_limit(text)

        async with self._request_semaphore:
            detected_lang = language or self._detect_language(text)
            text_norm, offset_mapper = await self._normalize_if_needed(text, detected_lang)

            explicit_mode = labels is not None or domains is not None
            label_groups = self._resolve_target_label_groups(
                text=text_norm,
                labels=labels,
                domains=domains,
                explicit_mode=explicit_mode,
            )

            if explicit_mode and not label_groups:
                entities: List[NerEntity] = []
                self._logger.info("Explicit labels/domains resolved to empty set; skipping inference.")
            else:
                entities = await self._infer_entities(text_norm, label_groups)

            entities = await self.enricher.enrich(entities, text, detected_lang)
            self._map_entities_back_to_original_text(entities, text, offset_mapper)

            final_entities = await self.resolver.resolve(entities, text, detected_lang)
            final_entities = self._apply_acceptance_policy(final_entities, domains=domains)
            final_entities = await self.qa.run_checks(final_entities, text)

        duration = (time.perf_counter() - start_time) * 1000
        return NerResult(
            language=detected_lang,
            entities=final_entities,
            usage={"char_count": len(text), "execution_time_ms": duration},
        )

    async def _normalize_if_needed(self, text: str, language: str) -> tuple[str, Any]:
        if not self.config.enable_normalization:
            return text, _IdentityOffsetMapper()
        return await self.normalizer.normalize(text, language)

    def _map_entities_back_to_original_text(self, entities: List[NerEntity], text: str, mapper: Any) -> None:
        for ent in entities:
            if self._is_model_entity(ent):
                orig_start, orig_end = mapper.map_back(ent.start, ent.end)
                ent.canonical_text = ent.text
                ent.start = max(orig_start, 0)
                ent.end = max(orig_end, ent.start)
                ent.text = text[ent.start:ent.end]
            else:
                ent.canonical_text = ent.text

    @staticmethod
    def _is_model_entity(entity: NerEntity) -> bool:
        source = (entity.source or "").lower()
        return source not in {"regex", "heuristic_split", "rule", "heuristic"}

    def _resolve_target_label_groups(
        self,
        *,
        text: str,
        labels: Optional[List[str] | Dict[str, str]],
        domains: Optional[List[str]],
        explicit_mode: bool,
    ) -> List[LabelSpec]:
        if explicit_mode:
            resolved = self._resolve_labels(labels, domains)
            return [resolved] if resolved else []

        groups = self._resolve_schema_label_groups(text, include_gated=True)
        if groups:
            return groups

        if self._default_label_groups:
            return self._default_label_groups

        if self.config.labels:
            return [dict(self.config.labels)]

        return []

    def _resolve_labels(
        self,
        labels: Optional[List[str] | Dict[str, str]],
        domains: Optional[List[str]],
    ) -> Optional[LabelSpec]:
        labels_as_map = labels if isinstance(labels, dict) else None
        label_items = self._normalize_items(list(labels_as_map.keys()) if labels_as_map else labels or [])
        domain_items = self._normalize_items(domains or [])
        cache_key = (tuple(label_items), tuple(domain_items)) if labels_as_map is None else None

        if cache_key is not None:
            cached = self._label_resolution_cache.get(cache_key)
            if cached is not None:
                return list(cached)

        resolved: set[str] = set()

        if domain_items:
            resolved_domains = self._resolve_domains(domain_items)
            for domain in resolved_domains:
                domain_cfg = self._domain_map.get(domain, {})
                for label in domain_cfg.get("labels", []):
                    label_name = str(label).strip().lower()
                    if label_name:
                        resolved.add(label_name)

        for label in label_items:
            mapped = self.label_resolver.resolve(label)
            if mapped:
                resolved.add(mapped)
            elif label in self._known_labels:
                resolved.add(label)

        if not resolved:
            if cache_key is not None:
                if len(self._label_resolution_cache) > 2048:
                    self._label_resolution_cache.clear()
                self._label_resolution_cache[cache_key] = None
            return None

        if labels_as_map is not None:
            resolved_map: Dict[str, str] = {}
            for raw_label, raw_description in labels_as_map.items():
                label_norm = str(raw_label).strip().lower()
                if not label_norm:
                    continue

                mapped = self.label_resolver.resolve(label_norm)
                target_label = mapped if mapped else (label_norm if label_norm in self._known_labels else None)
                if not target_label or target_label not in resolved:
                    continue

                description = str(raw_description).strip()
                if description:
                    resolved_map[target_label] = description

            for resolved_label in sorted(resolved):
                resolved_map.setdefault(resolved_label, resolved_label)
            return resolved_map

        final = sorted(resolved)
        if len(self._label_resolution_cache) > 2048:
            self._label_resolution_cache.clear()
        self._label_resolution_cache[cache_key] = final
        return final

    @staticmethod
    def _normalize_items(values: Sequence[str]) -> List[str]:
        return sorted({str(value).strip().lower() for value in values if str(value).strip()})

    def _resolve_domains(self, domain_items: Sequence[str]) -> List[str]:
        if not domain_items:
            return []

        resolved: List[str] = []
        for domain in domain_items:
            mapped = self.domain_resolver.resolve(domain)
            candidate = str(mapped).strip().lower() if mapped else domain
            if candidate in self._domain_map and candidate not in resolved:
                resolved.append(candidate)

        return resolved

    def _resolve_schema_label_groups(self, text: str, *, include_gated: bool) -> List[LabelSpec]:
        groups: List[LabelSpec] = []
        seen: set[tuple[str, ...]] = set()

        for pass_data in (self.config.schema_passes or {}).values():
            if include_gated and not self._is_pass_enabled(text, pass_data):
                continue

            labels = self._extract_pass_labels(pass_data)
            if not labels:
                continue

            key = tuple(labels.keys()) if isinstance(labels, dict) else tuple(labels)
            if key in seen:
                continue
            seen.add(key)
            groups.append(labels)

        return groups

    @staticmethod
    def _extract_pass_labels(pass_data: Any) -> LabelSpec:
        if isinstance(pass_data, dict):
            cleaned_map: Dict[str, str] = {}
            for raw_label, raw_description in pass_data.items():
                if raw_label == "enabled_if":
                    continue
                label_name = str(raw_label).strip().lower()
                if not label_name or label_name in cleaned_map:
                    continue

                description = str(raw_description).strip() if isinstance(raw_description, str) else ""
                cleaned_map[label_name] = description if description else label_name
            return cleaned_map
        elif isinstance(pass_data, (list, tuple, set)):
            labels = [str(item) for item in pass_data]
        else:
            labels = []

        cleaned: List[str] = []
        seen: set[str] = set()
        for label in labels:
            label_name = str(label).strip().lower()
            if label_name and label_name not in seen:
                cleaned.append(label_name)
                seen.add(label_name)
        return cleaned

    @staticmethod
    def _is_pass_enabled(text: str, pass_data: Any) -> bool:
        if not isinstance(pass_data, dict):
            return True

        gating = pass_data.get("enabled_if")
        if not gating:
            return True

        keywords = gating.get("any_keyword", [])
        if not keywords:
            return True

        text_lower = text.lower()
        return any(str(keyword).lower() in text_lower for keyword in keywords)

    async def _infer_entities(self, text: str, label_groups: List[LabelSpec]) -> List[NerEntity]:
        if not label_groups:
            return []

        segments = self._build_segments(text)
        if not segments:
            return []

        tasks = [self._predict_segments(segments, labels) for labels in label_groups]
        results = await asyncio.gather(*tasks)

        merged: List[NerEntity] = []
        for entities in results:
            merged.extend(entities)
        return merged

    def _build_segments(self, text: str) -> List[tuple[str, int]]:
        if self.config.split_lines:
            line_segments = self._split_lines(text)
            if line_segments:
                return line_segments

        chunk_cfg = (self.config.pipeline or {}).get("chunking", {})
        if chunk_cfg.get("enabled", True):
            max_chars = int(chunk_cfg.get("max_chars", 850))
            if len(text) > max_chars:
                return self._chunk_text(text, chunk_cfg)

        return [(text, 0)]

    @staticmethod
    def _split_lines(text: str) -> List[tuple[str, int]]:
        # Manual newline scan avoids allocating the full list produced by str.splitlines.
        segments: List[tuple[str, int]] = []
        start = 0
        index = 0
        length = len(text)

        while index < length:
            ch = text[index]
            if ch in "\r\n":
                line = text[start:index]
                if line.strip():
                    segments.append((line, start))

                if ch == "\r" and index + 1 < length and text[index + 1] == "\n":
                    index += 1
                start = index + 1
            index += 1

        if start < length:
            line = text[start:length]
            if line.strip():
                segments.append((line, start))

        return segments

    def _iter_segment_batches(self, segments: Sequence[tuple[str, int]]) -> Iterator[tuple[List[str], List[int]]]:
        batch_texts: List[str] = []
        batch_offsets: List[int] = []
        batch_chars = 0

        for text, offset in segments:
            if not text.strip():
                continue

            text_chars = len(text)
            should_flush = (
                bool(batch_texts)
                and (len(batch_texts) >= self._batch_size or (batch_chars + text_chars) > self._batch_max_chars)
            )
            if should_flush:
                yield batch_texts, batch_offsets
                batch_texts, batch_offsets, batch_chars = [], [], 0

            batch_texts.append(text)
            batch_offsets.append(offset)
            batch_chars += text_chars

            if len(batch_texts) >= self._batch_size or batch_chars >= self._batch_max_chars:
                yield batch_texts, batch_offsets
                batch_texts, batch_offsets, batch_chars = [], [], 0

        if batch_texts:
            yield batch_texts, batch_offsets

    @staticmethod
    def _chunk_text(text: str, config: Dict[str, Any]) -> List[tuple[str, int]]:
        max_chars = max(int(config.get("max_chars", 850)), 128)
        overlap = max(int(config.get("overlap_chars", 48)), 0)
        overlap = min(overlap, max_chars // 2)

        segments: List[tuple[str, int]] = []
        start = 0

        while start < len(text):
            end = min(start + max_chars, len(text))

            if end < len(text):
                boundary_candidates = [
                    text.rfind("\n", start + int(max_chars * 0.55), end),
                    text.rfind(". ", start + int(max_chars * 0.55), end),
                ]
                boundary = max(boundary_candidates)
                if boundary > start:
                    end = boundary + 1

            segment = text[start:end]
            if segment.strip():
                segments.append((segment, start))

            if end >= len(text):
                break

            next_start = end - overlap
            if next_start <= start:
                next_start = end
            start = next_start

        return segments

    async def _predict_segments(self, segments: List[tuple[str, int]], labels: LabelSpec) -> List[NerEntity]:
        pending: set[asyncio.Task[List[NerEntity]]] = set()
        entities: List[NerEntity] = []
        created_batches = False

        async def _drain_one() -> None:
            nonlocal pending
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                entities.extend(task.result())

        try:
            for texts, offsets in self._iter_segment_batches(segments):
                created_batches = True
                pending.add(asyncio.create_task(self._predict_batch_with_offsets(texts, offsets, labels)))
                if len(pending) >= self._max_inflight_batches:
                    await _drain_one()

            while pending:
                await _drain_one()
        except Exception:
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            raise

        if not created_batches:
            return []

        return entities

    async def _predict_batch_with_offsets(
        self,
        texts: List[str],
        offsets: List[int],
        labels: LabelSpec,
    ) -> List[NerEntity]:
        results = await self._predict_batch(texts, labels)
        entities: List[NerEntity] = []
        for text_entities, offset in zip(results, offsets):
            for ent in text_entities:
                ent.start += offset
                ent.end += offset
                entities.append(ent)
        return entities

    async def _predict_batch(self, texts: List[str], labels: LabelSpec) -> List[List[NerEntity]]:
        async with self._inference_semaphore:
            return await self.model.predict_batch(texts, labels)

    def _resolve_domain_threshold_policy(self, domains: Optional[List[str]]) -> Dict[str, Any]:
        if not domains:
            return {}

        threshold_cfg = self.config.thresholds or {}
        global_domain_overrides = threshold_cfg.get("domain_overrides", {}) or {}
        resolved_domains = self._resolve_domains(self._normalize_items(domains))
        if not resolved_domains:
            return {}

        merged: Dict[str, Any] = {}
        merged_overrides: Dict[str, Any] = {}

        for domain in resolved_domains:
            domain_configs = [
                global_domain_overrides.get(domain, {}),
                (self._domain_map.get(domain, {}) or {}).get("threshold_overrides", {}),
            ]
            for cfg in domain_configs:
                if not isinstance(cfg, dict):
                    continue
                if "high_accept" in cfg:
                    merged["high_accept"] = cfg["high_accept"]
                if "low_accept" in cfg:
                    merged["low_accept"] = cfg["low_accept"]

                overrides = cfg.get("overrides", {})
                if isinstance(overrides, dict):
                    merged_overrides.update(overrides)

        if merged_overrides:
            merged["overrides"] = merged_overrides

        return merged

    def _apply_acceptance_policy(self, entities: List[NerEntity], domains: Optional[List[str]] = None) -> List[NerEntity]:
        policy = self.config.thresholds or {}
        domain_policy = self._resolve_domain_threshold_policy(domains)

        high = float(domain_policy.get("high_accept", policy.get("high_accept", 0.85)))
        low = float(domain_policy.get("low_accept", policy.get("low_accept", 0.35)))
        overrides = dict(policy.get("overrides", {}) or {})
        overrides.update(domain_policy.get("overrides", {}) or {})

        accepted: List[NerEntity] = []
        for ent in entities:
            threshold = float(overrides.get(ent.label, low))
            if ent.score >= high or ent.score >= threshold:
                accepted.append(ent)
        return accepted

    @staticmethod
    def _detect_language(text: str) -> str:
        sample = text[:3000]
        if any("\u0590" <= char <= "\u05FF" for char in sample):
            return "he"
        if any("\u0600" <= char <= "\u06FF" for char in sample):
            if any(char in "یپچگک" for char in sample):
                return "fa"
            return "ar"
        return "en"

    async def classify(
        self,
        text: str,
        labels: Dict[str, List[str]],
        multi_label: bool = False,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        self._check_limit(text)

        async with self._request_semaphore:
            detected_lang = language or self._detect_language(text)
            results = await self.model.classify(text, labels, multi_label=multi_label)

        duration = (time.perf_counter() - start_time) * 1000
        return {
            "language": detected_lang,
            "results": results,
            "usage": {"char_count": len(text), "execution_time_ms": duration},
        }

    async def extract_json(
        self,
        text: str,
        schema: Dict[str, List[str]],
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        self._check_limit(text)

        async with self._request_semaphore:
            detected_lang = language or self._detect_language(text)
            data = await self.model.extract_json(text, schema)

        duration = (time.perf_counter() - start_time) * 1000
        return {
            "language": detected_lang,
            "data": data,
            "usage": {"char_count": len(text), "execution_time_ms": duration},
        }

    async def extract_relations(
        self,
        text: str,
        labels: List[str],
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        self._check_limit(text)

        async with self._request_semaphore:
            detected_lang = language or self._detect_language(text)
            relations = await self.model.extract_relations(text, labels)

        duration = (time.perf_counter() - start_time) * 1000
        return {
            "language": detected_lang,
            "relations": relations,
            "usage": {"char_count": len(text), "execution_time_ms": duration},
        }

    async def process_file(
        self,
        file_bytes: bytes,
        filename: str,
        *,
        language: Optional[str] = None,
        labels: Optional[List[str] | Dict[str, str]] = None,
        domains: Optional[List[str]] = None,
    ) -> NerResult:
        self._logger.info("Processing file: %s", filename)
        import anyio

        text = await anyio.to_thread.run_sync(self._extract_text, file_bytes, filename)
        return await self.process_text(text, language=language, labels=labels, domains=domains)

    async def process_upload_file(
        self,
        upload_file: Any,
        *,
        language: Optional[str] = None,
        labels: Optional[List[str] | Dict[str, str]] = None,
        domains: Optional[List[str]] = None,
    ) -> NerResult:
        filename = str(getattr(upload_file, "filename", "") or "upload.txt")
        self._logger.info("Processing uploaded file: %s", filename)
        suffix = filename.lower()

        if suffix.endswith(".txt"):
            text = await self._read_txt_upload(upload_file)
            return await self.process_text(text, language=language, labels=labels, domains=domains)
        if suffix.endswith(".pdf"):
            text = await self._read_pdf_upload(upload_file)
            return await self.process_text(text, language=language, labels=labels, domains=domains)

        file_bytes = await upload_file.read()
        return await self.process_file(file_bytes, filename, language=language, labels=labels, domains=domains)

    async def _read_txt_upload(self, upload_file: Any) -> str:
        import anyio

        raw_file = getattr(upload_file, "file", None)
        if raw_file is not None:
            return await anyio.to_thread.run_sync(self._decode_text_stream, raw_file)

        file_bytes = await upload_file.read()
        return file_bytes.decode("utf-8", errors="ignore")

    async def _read_pdf_upload(self, upload_file: Any) -> str:
        import anyio

        raw_file = getattr(upload_file, "file", None)
        if raw_file is not None:
            return await anyio.to_thread.run_sync(self._extract_pdf_text_stream, raw_file)

        file_bytes = await upload_file.read()
        return await anyio.to_thread.run_sync(self._extract_pdf_text_bytes, file_bytes)

    @staticmethod
    def _decode_text_stream(raw_file: Any, chunk_size: int = 1024 * 1024) -> str:
        import codecs

        try:
            raw_file.seek(0)
        except Exception:
            pass

        decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
        parts: List[str] = []

        while True:
            chunk = raw_file.read(chunk_size)
            if not chunk:
                break

            if isinstance(chunk, str):
                parts.append(chunk)
            else:
                parts.append(decoder.decode(chunk))

        parts.append(decoder.decode(b"", final=True))

        try:
            raw_file.seek(0)
        except Exception:
            pass

        return "".join(parts)

    @staticmethod
    def _extract_pdf_text_stream(raw_file: Any, chunk_size: int = 1024 * 1024) -> str:
        import os
        import tempfile

        try:
            import fitz
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF extraction. Install with 'pip install pymupdf'."
            ) from exc

        try:
            raw_file.seek(0)
        except Exception:
            pass

        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as temp_file:
                temp_path = temp_file.name
                while True:
                    chunk = raw_file.read(chunk_size)
                    if not chunk:
                        break
                    if isinstance(chunk, str):
                        temp_file.write(chunk.encode("utf-8", errors="ignore"))
                    else:
                        temp_file.write(chunk)

            with fitz.open(temp_path) as doc:
                pages: List[str] = []
                for page in doc:
                    text = page.get_text()
                    if text:
                        pages.append(text)
                return "\n".join(pages)
        finally:
            try:
                raw_file.seek(0)
            except Exception:
                pass
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    @staticmethod
    def _extract_pdf_text_bytes(file_bytes: bytes) -> str:
        try:
            import fitz
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF extraction. Install with 'pip install pymupdf'."
            ) from exc

        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            pages: List[str] = []
            for page in doc:
                text = page.get_text()
                if text:
                    pages.append(text)
            return "\n".join(pages)

    @staticmethod
    def _extract_text(file_bytes: bytes, filename: str) -> str:
        suffix = filename.lower()
        if suffix.endswith(".txt"):
            return file_bytes.decode("utf-8", errors="ignore")

        if suffix.endswith(".docx"):
            from io import BytesIO
            from docx import Document

            doc = Document(BytesIO(file_bytes))
            return "\n".join(paragraph.text for paragraph in doc.paragraphs)

        if suffix.endswith(".pdf"):
            return IEService._extract_pdf_text_bytes(file_bytes)

        raise ValueError("Unsupported file type; only .txt, .docx, and .pdf are supported")


NerService = IEService


def build_service(config_path: Path) -> IEService:
    config = NerConfig.from_yaml(config_path)
    model = GlinerModel(config)
    return IEService(config, model)


__all__ = [
    "NerResult",
    "IEService",
    "NerService",
    "build_service",
]
