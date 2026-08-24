from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import GEMINI_MODEL, MAX_IMAGE_SIZE_MB
from app.services.gemini_vision import vision_service

router = APIRouter(prefix="/api/v1/cv", tags=["Computer Vision"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    description: str | None = Form(None),
):
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, "Unsupported image type. Use JPEG, PNG or WEBP.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(400, "Empty image.")

    max_size = MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(image_bytes) > max_size:
        raise HTTPException(413, f"Image exceeds {MAX_IMAGE_SIZE_MB} MB limit.")

    try:
        result = await vision_service.analyze(
            image_bytes=image_bytes,
            mime_type=image.content_type,
            user_description=description,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, "Unexpected vision service failure") from exc

    return {
        "success": True,
        "provider": "gemini",
        "model": GEMINI_MODEL,
        "filename": image.filename,
        "analysis": result.model_dump(),
    }
