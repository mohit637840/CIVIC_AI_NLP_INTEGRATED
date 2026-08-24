from __future__ import annotations

import base64
import binascii
import logging
from urllib.parse import urlparse

import httpx

from app.schemas.submission import (
    DescriptionResult,
    ImageResult,
    SubmissionContext,
    SubmissionRequest,
    SubmissionResponse,
)
from app.schemas.fusion import FusionInput
from app.services.geo_service import geo_service
from app.services.gemini_vision import vision_service
from app.services.nlp_service import nlp_service
from app.services.fusion_service import fusion_service
from app.services.duplicate_service import duplicate_service
from app.services.priority_service import priority_service
from app.services.routing_service import routing_service


logger = logging.getLogger(__name__)


async def _load_image(payload: SubmissionRequest) -> tuple[bytes, str] | None:
    if not payload.image_base64 and not payload.image_url:
        return None

    if bool(payload.image_base64) == bool(payload.image_url):
        raise ValueError("Provide exactly one of image_base64 or image_url")

    if payload.image_base64:
        raw = payload.image_base64
        mime = "image/jpeg"
        if raw.startswith("data:"):
            header, raw = raw.split(",", 1)
            mime = header[5:].split(";", 1)[0] or mime
        try:
            return base64.b64decode(raw, validate=True), mime
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Invalid base64 image") from exc

    parsed = urlparse(payload.image_url or "")
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("image_url must be an http(s) URL")

    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        response = await client.get(payload.image_url)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "image/jpeg").split(";", 1)[0]
    return response.content, content_type


async def _resolve_location(payload: SubmissionRequest):
    if payload.location is None:
        return None

    if payload.location.gps_coordinates:
        p = payload.location.gps_coordinates
        return await geo_service.reverse_geocode(p.lat, p.lng)

    assert payload.location.manual_address is not None
    return await geo_service.forward_geocode(payload.location.manual_address.text)


async def build_submission(payload: SubmissionRequest) -> SubmissionResponse:
    submission_id = payload.request_id
    logger.info("submission_id=%s pipeline_started", submission_id)
    # Run only the modalities that the citizen supplied.
    nlp_result = None
    nlp_error = None
    if payload.description:
        logger.info("submission_id=%s nlp_started", submission_id)
        try:
            nlp_result = nlp_service.analyze(payload.description)
            logger.info("submission_id=%s nlp_completed", submission_id)
        except Exception:
            nlp_error = "NLP analysis is currently unavailable"

    image_result = None
    image_error = None
    image_payload = await _load_image(payload)
    if image_payload is not None:
        logger.info("submission_id=%s cv_started", submission_id)
        image_bytes, mime_type = image_payload
        try:
            image_result = await vision_service.analyze(
                image_bytes=image_bytes,
                mime_type=mime_type,
                user_description=payload.description,
            )
            logger.info("submission_id=%s cv_completed", submission_id)
        except Exception:
            image_error = "Computer vision analysis is currently unavailable"

    location = None
    location_error = None
    try:
        logger.info("submission_id=%s geo_started", submission_id)
        location = await _resolve_location(payload)
        logger.info("submission_id=%s geo_completed", submission_id)
    except Exception:
        location_error = "Location resolution is currently unavailable"

    logger.info("submission_id=%s fusion_started", submission_id)
    fusion = fusion_service.build(FusionInput(
        nlp_result=nlp_result,
        vision_result=image_result,
        geo_result=location,
        nlp_error=nlp_error,
        vision_error=image_error,
        geo_error=location_error,
    ))
    logger.info("submission_id=%s fusion_completed", submission_id)

    context = SubmissionContext(
        id=submission_id,
        description=DescriptionResult(
            available=nlp_result is not None,
            analysis=nlp_result,
            error=nlp_error,
        ),
        image=ImageResult(
            available=image_result is not None,
            provider="gemini" if image_result is not None else None,
            model=vision_service.model if image_result is not None else None,
            analysis=image_result,
            error=image_error,
        ),
        location=location,
        location_error=location_error,
        # Deliberately left empty until the three-input rule-based fusion layer
        # is implemented against the real NLP/CV/Geo outputs.
        fusion=fusion,
        duplicate=None,
        priority=None,
        routing=None,
    )
    logger.info("submission_id=%s duplicate_started", submission_id)
    context = duplicate_service.process(context)
    logger.info("submission_id=%s duplicate_completed", submission_id)
    logger.info("submission_id=%s priority_started", submission_id)
    context = priority_service.process(context)
    logger.info("submission_id=%s priority_completed", submission_id)
    logger.info("submission_id=%s routing_started", submission_id)
    context = routing_service.process(context)
    logger.info("submission_id=%s routing_completed", submission_id)
    logger.info("submission_id=%s pipeline_completed", submission_id)

    return SubmissionResponse(**context.model_dump())
