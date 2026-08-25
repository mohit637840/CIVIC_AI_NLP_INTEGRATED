from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DuplicateEvidence(BaseModel):
    raw_complaint: str | None = None
    normalized_complaint: str | None = None
    canonical_complaint: str | None = None
    summary: str | None = None
    problem_type: str | None = None
    category: str | None = None
    domain: str | None = None
    issue_type: str | None = None
    landmark: str | None = None
    textual_location: str | None = None
    cause: str | None = None
    affected_people: list[str] = Field(default_factory=list)
    urgency: str | None = None
    severity: str | None = None


class DuplicateCaseRepresentation(BaseModel):
    normalized_complaint: str | None = None
    canonical_complaint: str | None = None
    issue_focused_text: str | None = None
    category: str | None = None
    domain: str | None = None
    issue_type: str | None = None
    landmark: str | None = None
    textual_location: str | None = None
    visual_domain: str | None = None
    visual_issue_type: str | None = None
    visual_description: str | None = None
    visible_conditions: list[str] = Field(default_factory=list)
    affected_objects: list[str] = Field(default_factory=list)
    hazards: list[str] = Field(default_factory=list)
    environmental_indicators: list[str] = Field(default_factory=list)
    obstruction_indicators: list[str] = Field(default_factory=list)
    latitude: float | None = None
    longitude: float | None = None
    district: str | None = None
    block: str | None = None
    state: str | None = None
    formatted_address: str | None = None
    submission_timestamp: str | None = None
    duration_minutes: float | None = None
    severity: str | None = None
    urgency: str | None = None
    affected_people: list[str] = Field(default_factory=list)
    public_impact: str | None = None


class DuplicateSignalBreakdown(BaseModel):
    semantic_similarity: float | None = None
    text_vector_similarity: float | None = None
    lexical_similarity: float | None = None
    category_compatibility: float | None = None
    issue_compatibility: float | None = None
    domain_compatibility: float | None = None
    geo_distance_meters: float | None = None
    geo_similarity: float | None = None
    same_district: bool | None = None
    same_block: bool | None = None
    same_landmark: bool | None = None
    temporal_similarity: float | None = None
    visual_evidence_compatibility: float | None = None
    visual_similarity: float | None = None


class CandidateEvidence(BaseModel):
    submission_id: str
    rank: int
    retrieval_score: float | None = None
    ranking_score: float | None = None
    decision_score: float | None = None
    score: float | None = None
    retrieved_by: list[str] = Field(default_factory=list)
    signals: DuplicateSignalBreakdown
    explanation: list[str] = Field(default_factory=list)


class DuplicateDecision(BaseModel):
    threshold: float
    score: float | None = None
    decision: str
    reason: str


class DuplicateResult(BaseModel):
    status: str
    is_duplicate: bool = False
    best_match: dict[str, Any] | None = None
    candidates: list[CandidateEvidence] = Field(default_factory=list)
    signals: DuplicateSignalBreakdown
    signal_breakdown: DuplicateSignalBreakdown | None = None
    positive_evidence: list[str] = Field(default_factory=list)
    negative_evidence: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    decision: DuplicateDecision
    explanation: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    representation: DuplicateCaseRepresentation | None = None
