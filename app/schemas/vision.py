from typing import Literal

from pydantic import BaseModel, Field


Severity = Literal[
    "low",
    "medium",
    "high",
    "critical",
    "unknown",
]


class VisualIssue(BaseModel):
    visual_domain: str = Field(
        description=(
            "Broad civic domain visible in the image, such as "
            "Waste Management, Water / Drainage, Road Infrastructure, "
            "Electricity, Environment, Public Infrastructure, "
            "Healthcare, Agriculture, Education, or Other."
        )
    )

    visual_issue_type: str = Field(
        description=(
            "Specific problem visually identified in the image."
        )
    )

    visual_severity: Severity = Field(
        description=(
            "Severity estimated only from visual evidence."
        )
    )

    visual_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Model-reported confidence in the visual identification."
        )
    )

    visual_description: str = Field(
        description=(
            "Detailed description of what is visibly happening."
        )
    )

    visual_evidence: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete observations from the image supporting "
            "the identified civic issue."
        )
    )

    affected_objects: list[str] = Field(
        default_factory=list,
        description=(
            "Objects, infrastructure, people, vehicles, or public "
            "spaces visibly affected."
        )
    )

    hazard_indicators: list[str] = Field(
        default_factory=list,
        description=(
            "Visible indicators of danger or public safety risk."
        )
    )

    environmental_indicators: list[str] = Field(
        default_factory=list,
        description=(
            "Visible environmental or sanitation concerns."
        )
    )

    obstruction_indicators: list[str] = Field(
        default_factory=list,
        description=(
            "Visible obstruction of roads, sidewalks, drains, "
            "public spaces, etc."
        )
    )

    visible_scale: str = Field(
        default="unknown",
        description=(
            "Approximate visible scale of the issue: "
            "small, localized, widespread, or unknown."
        )
    )

    visible_conditions: list[str] = Field(
        default_factory=list,
        description=(
            "Other relevant visible conditions such as standing water, "
            "scattered waste, damaged surfaces, exposed wires, etc."
        )
    )

    estimated_public_impact: str = Field(
        description=(
            "Likely impact on citizens/public infrastructure based "
            "only on visual evidence."
        )
    )

    recommended_visual_action: str = Field(
        description=(
            "Suggested action based on the visible condition. "
            "Do not invent administrative information."
        )
    )


class VisionAnalysisResult(BaseModel):
    image_valid: bool

    issue_detected: bool

    issues: list[VisualIssue] = Field(
        default_factory=list
    )

    image_quality_notes: str = ""

    visible_context: list[str] = Field(
        default_factory=list,
        description=(
            "General useful visual context that is not itself "
            "necessarily a civic issue."
        )
    )

    error: str | None = None