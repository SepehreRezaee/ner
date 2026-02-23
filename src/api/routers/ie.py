from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException
from ..dependencies_ner import get_ner_service
from src.engines.nlp.ner import IEService
from schemas.ie import (
    ClassificationRequest, ClassificationResponse,
    JsonExtractionRequest, JsonExtractionResponse,
    RelationExtractionRequest, RelationExtractionResponse
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ie", tags=["ie"])

@router.post("/classify", response_model=ClassificationResponse)
async def classify_text(
    request: ClassificationRequest,
    service: IEService = Depends(get_ner_service)
) -> ClassificationResponse:
    logger.info("IE: classify endpoint called")
    try:
        data = await service.classify(request.text, request.labels, request.multi_label, language=request.language)
        return ClassificationResponse(**data)
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except Exception as exc:
        logger.error("Error in IE classify: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

@router.post("/extract-json", response_model=JsonExtractionResponse)
async def extract_json(
    request: JsonExtractionRequest,
    service: IEService = Depends(get_ner_service)
) -> JsonExtractionResponse:
    logger.info("IE: extract-json endpoint called")
    try:
        data = await service.extract_json(request.text, request.schema_def, language=request.language)
        return JsonExtractionResponse(**data)
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except Exception as exc:
        logger.error("Error in IE extract-json: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

@router.post("/extract-relations", response_model=RelationExtractionResponse)
async def extract_relations(
    request: RelationExtractionRequest,
    service: IEService = Depends(get_ner_service)
) -> RelationExtractionResponse:
    logger.info("IE: extract-relations endpoint called")
    try:
        data = await service.extract_relations(request.text, request.labels, language=request.language)
        return RelationExtractionResponse(**data)
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except Exception as exc:
        logger.error("Error in IE extract-relations: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
