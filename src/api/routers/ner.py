from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Form, Request, Query

from ..dependencies_ner import get_ner_service
from src.engines.nlp.ner import NerEntity, NerService
from schemas.ner import NerEntitySchema, NerResponse

router = APIRouter(prefix="/ner", tags=["ner"])


def _to_schema(entity: NerEntity) -> NerEntitySchema:
    return NerEntitySchema(
        text=entity.text,
        label=entity.label,
        start=entity.start,
        end=entity.end,
        score=entity.score,
        source=entity.source,
    )


def _to_optional_str_list(value: object) -> list[str] | None:
    if not isinstance(value, list):
        return None
    return [str(item) for item in value]


@router.post("/process-text", response_model=NerResponse)
async def process_text(
    request: Request,
    text: str | None = Form(default=None, alias="text"),
    language_form: str | None = Form(default=None, alias="language"),
    language_query: str | None = Query(default=None, alias="language"),
    labels_query: list[str] | None = Query(default=None, alias="labels"),
    domains_query: list[str] | None = Query(default=None, alias="domain"),
    service: NerService = Depends(get_ner_service),
) -> NerResponse:
    resolved_text = text
    resolved_language = language_form or language_query
    resolved_labels = labels_query
    resolved_domains = domains_query

    if resolved_text is None:
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        if isinstance(payload, dict):
            resolved_text = payload.get("text")
            resolved_language = payload.get("language", resolved_language)
            payload_labels = _to_optional_str_list(payload.get("labels"))
            if payload_labels is not None:
                resolved_labels = payload_labels
            payload_domains = _to_optional_str_list(payload.get("domain")) or _to_optional_str_list(payload.get("domains"))
            if payload_domains is not None:
                resolved_domains = payload_domains

    if not resolved_text:
        raise HTTPException(status_code=422, detail="Provide JSON body or form-data with text")

    try:
        result = await service.process_text(
            resolved_text,
            language=resolved_language,
            labels=resolved_labels,
            domains=resolved_domains,
        )
    except RuntimeError as exc:
        if "Rate limit exceeded" in str(exc):
            raise HTTPException(status_code=429, detail=str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return NerResponse(
        language=result.language,
        entities=[_to_schema(e) for e in result.entities],
        usage=result.usage,
    )


@router.post("/process-file", response_model=NerResponse)
async def process_file(
    file: UploadFile = File(...),
    language: str | None = None,
    labels: list[str] | None = Query(default=None),
    domain: list[str] | None = Query(default=None),
    service: NerService = Depends(get_ner_service),
) -> NerResponse:
    try:
        result = await service.process_upload_file(
            file,
            language=language,
            labels=labels,
            domains=domain,
        )
        return NerResponse(
            language=result.language,
            entities=[_to_schema(e) for e in result.entities],
            usage=result.usage,
        )
    except RuntimeError as exc:
        if "Rate limit exceeded" in str(exc):
            raise HTTPException(status_code=429, detail=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


__all__ = ["router"]
