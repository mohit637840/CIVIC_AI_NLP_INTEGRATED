from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.geo import ResolvedLocation
from app.schemas.nlp import NLPAnalysis
from app.schemas.vision import VisionAnalysisResult


class FusionInput(BaseModel):
    nlp_result: NLPAnalysis | None = None
    vision_result: VisionAnalysisResult | None = None
    geo_result: ResolvedLocation | None = None
    nlp_error: str | None = None
    vision_error: str | None = None
    geo_error: str | None = None


class FusedCategory(BaseModel):
    value: str | None = None
    issue_type: str | None = None
    domain: str | None = None
    source: str | None = None
    confidence: float | None = None
    reason: str | None = None


class FusedSeverity(BaseModel):
    value: str | None = None
    source: str | None = None
    confidence: float | None = None
    reason: str | None = None


class FusedLocation(BaseModel):
    lat: float
    lng: float
    district: str | None = None
    block: str | None = None
    state: str | None = None
    formatted_address: str | None = None
    source: str
    resolution_status: str
    confidence: float | None = None


class LocationContext(BaseModel):
    nlp_location: str | None = None
    nlp_landmark: str | None = None
    geo_authoritative: bool = False


class ImpactSummary(BaseModel):
    affected_people: list[str] = Field(default_factory=list)
    affected_objects: list[str] = Field(default_factory=list)
    public_impact: str | None = None
    scale: str | None = None


class Evidence(BaseModel):
    text: list[str] = Field(default_factory=list)
    vision: list[str] = Field(default_factory=list)
    geo: list[str] = Field(default_factory=list)


class ModalityAvailability(BaseModel):
    nlp: bool = False
    vision: bool = False
    geo: bool = False


class ModalityState(BaseModel):
    available: bool = False
    usable: bool = False
    status: str = "not_provided"
    error: str | None = None


class ModalityStates(BaseModel):
    nlp: ModalityState
    vision: ModalityState
    geo: ModalityState


class ConfidenceSummary(BaseModel):
    overall: float | None = None
    category: float | None = None
    severity: float | None = None
    location: float | None = None


class FusionConflict(BaseModel):
    field: str
    type: str = "modality_conflict"
    detected: bool = True
    nlp_value: Any = None
    vision_value: Any = None
    geo_value: Any = None
    selected_value: Any = None
    selected_source: str | None = None
    resolution: str
    reason: str


class FusionDecision(BaseModel):
    field: str
    selected_source: str | None = None
    considered_sources: list[str] = Field(default_factory=list)
    conflict: bool = False
    reason: str


class FusedPriority(BaseModel):
    value: str | None = None
    source: str = "rule_engine"
    reason: str
    factors: list[str] = Field(default_factory=list)


class FusedDepartment(BaseModel):
    value: str | None = None
    source: str = "category_mapping"
    reason: str


class FusedRouting(BaseModel):
    department: FusedDepartment
    reason: str


class DuplicateFeatures(BaseModel):
    category: str | None = None
    issue_type: str | None = None
    domain: str | None = None
    normalized_complaint: str | None = None
    visual_description: str | None = None
    relevant_evidence: list[str] = Field(default_factory=list)
    coordinates: dict[str, float] | None = None
    district: str | None = None
    block: str | None = None
    state: str | None = None
    severity: str | None = None
    landmark: str | None = None
    affected_objects: list[str] = Field(default_factory=list)


class ExplainabilityMetadata(BaseModel):
    engine: str = "rule_based"
    version: str = "1.0"
    vision_category_confidence_threshold: float
    vision_severity_confidence_threshold: float
    rules: list[str] = Field(default_factory=list)


class FusionResult(BaseModel):
    fusion_status: str
    category: FusedCategory
    severity: FusedSeverity
    priority_candidate: str | None = None
    priority: FusedPriority
    department: FusedDepartment
    routing: FusedRouting
    location: FusedLocation | None = None
    location_context: LocationContext
    impact: ImpactSummary
    hazards: list[str] = Field(default_factory=list)
    environmental_indicators: list[str] = Field(default_factory=list)
    obstruction_indicators: list[str] = Field(default_factory=list)
    visible_conditions: list[str] = Field(default_factory=list)
    recommended_action: dict[str, str | None]
    evidence: Evidence
    modality_availability: ModalityAvailability
    modality_states: ModalityStates
    confidence: ConfidenceSummary
    conflicts: list[FusionConflict] = Field(default_factory=list)
    decisions: list[FusionDecision] = Field(default_factory=list)
    explainability: ExplainabilityMetadata
    duplicate_features: DuplicateFeatures