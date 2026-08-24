from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class NLPAnalysis(BaseModel):
    """Structured contract matching the current teammate NLP output."""

    model_config = ConfigDict(extra="allow")

    complaint: str
    language: str | None = None
    primary_category: str | None = None
    secondary_category: str | None = None
    problem_type: str | None = None
    severity: str | None = None
    priority: str | None = None
    location: str | None = None
    landmark: str | None = None
    duration: str | None = None
    cause: str | None = None
    affected_people: Any = None
    urgency: str | None = None
    department: str | None = None
    suggested_action: str | None = None
    domain: str | None = None
    confidence: float | None = None
