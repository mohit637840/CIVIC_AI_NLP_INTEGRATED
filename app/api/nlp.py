from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.nlp_service import nlp_service

router = APIRouter(prefix="/api/v1/nlp", tags=["NLP"])


class NLPAnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)


@router.post("/analyze")
def analyze_text(payload: NLPAnalyzeRequest) -> dict:
    """Run the teammate's existing NLP pipeline on complaint text."""
    try:
        return {
            "success": True,
            "analysis": nlp_service.analyze(payload.text),
        }
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, "NLP model failed to analyze the complaint") from exc
