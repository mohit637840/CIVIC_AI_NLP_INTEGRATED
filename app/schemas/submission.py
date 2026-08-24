from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from app.schemas.geo import LocationInput, ResolvedLocation
from app.schemas.nlp import NLPAnalysis
from app.schemas.vision import VisionAnalysisResult
from app.schemas.fusion import FusedPriority, FusedRouting, FusionResult


class DescriptionResult(BaseModel):
    available: bool
    analysis: NLPAnalysis | None = None
    error: str | None = None


class ImageResult(BaseModel):
    available: bool
    provider: str | None = None
    model: str | None = None
    analysis: VisionAnalysisResult | None = None
    error: str | None = None


class SubmissionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "description": "There is severe waterlogging on the road near Main Market in Ranchi.",
                "image_base64": None,
                "image_url": None,
                "location": {
                    "gps_coordinates": {"lat": 23.3441, "lng": 85.3096}
                },
            }
        }
    )

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    description: str | None = Field(
        default=None,
        min_length=1,
        validation_alias=AliasChoices("description", "description_text"),
    )
    image_base64: str | None = None
    image_url: str | None = None
    location: LocationInput | None = None

    @model_validator(mode="after")
    def validate_sources(self) -> "SubmissionRequest":
        if bool(self.image_base64) and bool(self.image_url):
            raise ValueError("Provide only one of image_base64 or image_url")

        if not any(
            (
                self.description and self.description.strip(),
                self.image_base64,
                self.image_url,
                self.location,
            )
        ):
            raise ValueError("Provide at least one of description, image, or location")

        return self


class SubmissionContext(BaseModel):
    """Internal canonical case passed between post-fusion processors."""

    id: str
    description: DescriptionResult
    image: ImageResult
    location: ResolvedLocation | None = None
    location_error: str | None = None
    fusion: FusionResult
    duplicate: dict[str, Any] | None = None
    priority: FusedPriority | None = None
    routing: FusedRouting | None = None
    status: str = "pending_review"


class SubmissionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "submission-123",
                "description": {
                    "available": True,
                    "analysis": {
                        "complaint": "There is severe waterlogging near Main Market in Ranchi.",
                        "language": "English",
                        "primary_category": "Water Management",
                        "severity": "High",
                        "priority": "High",
                        "domain": "Water Management",
                    },
                    "error": None,
                },
                "image": {
                    "available": False,
                    "provider": None,
                    "model": None,
                    "analysis": None,
                    "error": None,
                },
                "location": {
                    "lat": 23.3441,
                    "lng": 85.3096,
                    "source": "gps",
                    "raw_input": "23.3441, 85.3096",
                    "district": "Ranchi",
                    "block": None,
                    "state": "Jharkhand",
                    "formatted_address": None,
                },
                "location_error": None,
                "fusion": {
                    "fusion_status": "complete",
                    "category": {
                        "value": "Water / Drainage",
                        "issue_type": "Waterlogging / Flooding",
                        "domain": "Water / Drainage",
                        "source": "vision",
                        "confidence": 0.95,
                        "reason": "Vision identified the issue above the configured confidence threshold.",
                    },
                    "severity": {
                        "value": "HIGH",
                        "source": "vision",
                        "confidence": 0.95,
                        "reason": "The highest normalized severity across available modalities was selected.",
                    },
                    "priority": {
                        "value": "HIGH",
                        "source": "rule_engine",
                        "reason": "Priority is derived from final fused severity and available impact factors.",
                        "factors": ["final severity is HIGH", "visual hazards detected"],
                    },
                    "department": {
                        "value": None,
                        "source": "category_mapping",
                        "reason": "No authoritative department mapping is available for the final category.",
                    },
                    "routing": {
                        "department": {
                            "value": None,
                            "source": "category_mapping",
                            "reason": "No authoritative department mapping is available for the final category.",
                        },
                        "reason": "No authoritative department mapping is available for the final category.",
                    },
                    "priority_candidate": "HIGH",
                    "location": {
                        "lat": 23.3441,
                        "lng": 85.3096,
                        "district": "Ranchi",
                        "block": None,
                        "state": "Jharkhand",
                        "formatted_address": None,
                        "source": "gps",
                        "resolution_status": "reverse_geocoded",
                        "confidence": 1.0,
                    },
                    "location_context": {
                        "nlp_location": "Main Market in Ranchi",
                        "nlp_landmark": "Main Market",
                        "geo_authoritative": True,
                    },
                    "impact": {
                        "affected_people": ["Residents", "Commuters"],
                        "affected_objects": ["Road infrastructure"],
                        "public_impact": "Traffic is disrupted.",
                        "scale": "widespread",
                    },
                    "hazards": ["Vehicle damage risk"],
                    "environmental_indicators": ["Standing water"],
                    "obstruction_indicators": ["Road obstruction"],
                    "visible_conditions": ["Deep water"],
                    "recommended_action": {
                        "action": "Inspect drainage and remove blockage.",
                        "source": "vision",
                    },
                    "evidence": {
                        "text": ["There is severe waterlogging near Main Market in Ranchi."],
                        "vision": ["Road covered by standing water"],
                        "geo": ["GPS coordinates resolved to Ranchi."],
                    },
                    "modality_availability": {"nlp": True, "vision": True, "geo": True},
                    "modality_states": {
                        "nlp": {"available": True, "usable": True, "status": "valid", "error": None},
                        "vision": {"available": True, "usable": True, "status": "valid", "error": None},
                        "geo": {"available": True, "usable": True, "status": "valid", "error": None},
                    },
                    "confidence": {
                        "overall": 0.95,
                        "category": 0.95,
                        "severity": 0.95,
                        "location": 1.0,
                    },
                    "conflicts": [],
                    "decisions": [
                        {
                            "field": "category",
                            "selected_source": "vision",
                            "reason": "Vision confidence exceeded configured threshold.",
                        },
                        {
                            "field": "location",
                            "selected_source": "geo",
                            "reason": "Resolved geospatial coordinates are authoritative.",
                        },
                    ],
                    "explainability": {
                        "engine": "rule_based",
                        "version": "1.0",
                        "vision_category_confidence_threshold": 0.75,
                        "vision_severity_confidence_threshold": 0.75,
                        "rules": ["vision_category_threshold", "geo_authority", "highest_reliable_severity", "priority_from_fused_evidence"],
                    },
                    "duplicate_features": {
                        "category": "Water / Drainage",
                        "issue_type": "Waterlogging / Flooding",
                        "domain": "Water / Drainage",
                        "normalized_complaint": "there is severe waterlogging near main market in ranchi.",
                        "visual_description": "Road completely submerged.",
                        "relevant_evidence": ["Severe waterlogging near Main Market"],
                        "coordinates": {"lat": 23.3441, "lng": 85.3096},
                        "district": "Ranchi",
                        "block": None,
                        "state": "Jharkhand",
                        "severity": "HIGH",
                        "landmark": "Main Market",
                        "affected_objects": ["Road infrastructure"],
                    },
                },
                "duplicate": None,
                "priority": {
                    "value": "HIGH",
                    "source": "rule_engine",
                    "reason": "Priority is derived from final fused severity and available impact factors.",
                    "factors": ["final severity is HIGH", "visual hazards detected"],
                },
                "routing": {
                    "department": {
                        "value": "Water Supply Department",
                        "source": "category_mapping",
                        "reason": "Mapped from the final fused category.",
                    },
                    "reason": "Mapped from the final fused category.",
                },
                "status": "pending_review",
            }
        }
    )

    id: str
    description: DescriptionResult
    image: ImageResult
    location: ResolvedLocation | None = None
    location_error: str | None = None
    fusion: FusionResult | None = None
    duplicate: dict[str, Any] | None = None
    priority: FusedPriority | None = None
    routing: FusedRouting | None = None
    status: str = "pending_review"
