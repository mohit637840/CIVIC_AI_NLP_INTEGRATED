from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.submission import SubmissionRequest, SubmissionResponse
from app.services.geo_service import GeoServiceError
from app.services.submission_service import build_submission

router = APIRouter(prefix="/api/v1/submissions", tags=["Submissions"])


@router.post("", response_model=SubmissionResponse)
async def create_submission(payload: SubmissionRequest) -> SubmissionResponse:
    """Run the supplied NLP, vision, and geospatial modalities for one submission."""
    try:
        return await build_submission(payload)
    except GeoServiceError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, "Submission processing failed") from exc
