from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from app.config import (
    VISION_CATEGORY_CONFIDENCE_THRESHOLD,
    VISION_SEVERITY_CONFIDENCE_THRESHOLD,
)
from app.schemas.fusion import (
    ConfidenceSummary,
    DuplicateFeatures,
    Evidence,
    ExplainabilityMetadata,
    FusedCategory,
    FusedDepartment,
    FusedLocation,
    FusedPriority,
    FusedRouting,
    FusedSeverity,
    FusionConflict,
    FusionDecision,
    FusionInput,
    FusionResult,
    ImpactSummary,
    LocationContext,
    ModalityAvailability,
    ModalityState,
    ModalityStates,
)
from app.schemas.vision import VisualIssue

logger = logging.getLogger(__name__)
SEVERITY_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
CATEGORY_ALIASES = {
    "water management": "water / drainage",
    "water/drainage": "water / drainage",
    "drainage": "water / drainage",
    "waterlogging / flooding": "water / drainage",
}
DEPARTMENT_BY_CATEGORY = {
    "Accessibility": "Social Welfare / Municipal Department",
    "Agriculture": "Agriculture Department",
    "Education": "Education Department",
    "Environment": "Environment Department",
    "Garbage & Waste Management": "Sanitation Department",
    "Healthcare": "Health Department",
    "Public Administration": "District Administration",
    "Public Services": "Relevant Public Service Department",
    "Road & Urban Infrastructure": "Road / Municipal Department",
    "Rural Livelihoods": "Rural Development Department",
    "Street Light & Energy": "Municipal / Electrical Department",
        "Water Management": "Water Supply Department",
        "Water / Drainage": "Water Supply Department",
    "Urban Development": "Urban Development Department",
}


def normalize_severity(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().upper()
    return value if value in SEVERITY_RANK else None


def normalize_term(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"\s+", " ", value.strip().lower())
    return CATEGORY_ALIASES.get(cleaned, cleaned)


def unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def vision_issue(result) -> VisualIssue | None:
    if not result or not result.image_valid or not result.issue_detected or not result.issues:
        return None
    candidates = [issue for issue in result.issues if issue.visual_issue_type and issue.visual_domain]
    return max(candidates, key=lambda issue: issue.visual_confidence) if candidates else None


class FusionService:
    def __init__(
        self,
        vision_category_confidence_threshold: float | None = None,
        vision_severity_confidence_threshold: float | None = None,
    ) -> None:
        self.category_threshold = (
            VISION_CATEGORY_CONFIDENCE_THRESHOLD
            if vision_category_confidence_threshold is None
            else vision_category_confidence_threshold
        )
        self.severity_threshold = (
            VISION_SEVERITY_CONFIDENCE_THRESHOLD
            if vision_severity_confidence_threshold is None
            else vision_severity_confidence_threshold
        )

    def modality_states(self, inputs: FusionInput) -> ModalityStates:
        nlp_state = ModalityState(
            available=inputs.nlp_result is not None,
            usable=inputs.nlp_result is not None and bool(inputs.nlp_result.complaint),
            status="valid" if inputs.nlp_result else "failed" if inputs.nlp_error else "not_provided",
            error=inputs.nlp_error,
        )
        visual = vision_issue(inputs.vision_result)
        if inputs.vision_result is None:
            vision_state = ModalityState(status="failed" if inputs.vision_error else "not_provided", error=inputs.vision_error)
        elif visual is None:
            vision_state = ModalityState(available=True, usable=False, status="invalid", error=inputs.vision_error)
        elif visual.visual_confidence < 0.5:
            vision_state = ModalityState(available=True, usable=True, status="low_confidence")
        else:
            vision_state = ModalityState(available=True, usable=True, status="valid")
        geo_state = ModalityState(
            available=inputs.geo_result is not None,
            usable=inputs.geo_result is not None,
            status="valid" if inputs.geo_result else "failed" if inputs.geo_error else "not_provided",
            error=inputs.geo_error,
        )
        return ModalityStates(nlp=nlp_state, vision=vision_state, geo=geo_state)

    def resolve_category(self, inputs: FusionInput) -> tuple[FusedCategory, FusionDecision]:
        nlp = inputs.nlp_result
        visual = vision_issue(inputs.vision_result)
        nlp_value = nlp.primary_category if nlp else None
        nlp_issue = nlp.problem_type if nlp else None
        if visual and visual.visual_confidence >= self.category_threshold:
            return FusedCategory(
                value=visual.visual_domain,
                issue_type=visual.visual_issue_type,
                domain=visual.visual_domain,
                source="vision",
                confidence=visual.visual_confidence,
                reason=f"Vision confidence {visual.visual_confidence:.2f} exceeded threshold {self.category_threshold:.2f}.",
            ), FusionDecision(
                field="category", selected_source="vision", considered_sources=["vision", "nlp"],
                conflict=bool(nlp_value and normalize_term(nlp_value) != normalize_term(visual.visual_domain)),
                reason="Vision confidence exceeded the configured category threshold.",
            )
        if visual and nlp and normalize_term(nlp_value) == normalize_term(visual.visual_domain) and normalize_term(nlp_issue) == normalize_term(visual.visual_issue_type):
            return FusedCategory(
                value=nlp_value, issue_type=nlp_issue, domain=nlp.domain, source="fused",
                confidence=visual.visual_confidence,
                reason="NLP and vision classifications agree after normalization.",
            ), FusionDecision(
                field="category", selected_source="fused", considered_sources=["vision", "nlp"], reason="Classifications agree after normalization."
            )
        if nlp:
            return FusedCategory(
                value=nlp_value, issue_type=nlp_issue, domain=nlp.domain, source="nlp",
                confidence=nlp.confidence,
                reason="NLP is the fallback because vision was unavailable or below threshold.",
            ), FusionDecision(
                field="category", selected_source="nlp", considered_sources=["vision", "nlp"],
                conflict=bool(visual and nlp_value), reason="Vision confidence did not meet the category threshold."
            )
        if visual:
            return FusedCategory(
                value=visual.visual_domain, issue_type=visual.visual_issue_type, domain=visual.visual_domain,
                source="vision", confidence=visual.visual_confidence,
                reason="Vision supplied the only usable classification evidence.",
            ), FusionDecision(field="category", selected_source="vision", considered_sources=["vision"], reason="Vision supplied the only usable classification evidence.")
        return FusedCategory(source="unresolved", reason="No usable classification evidence was available."), FusionDecision(field="category", reason="No usable classification evidence was available.")

    def resolve_severity(self, inputs: FusionInput) -> tuple[FusedSeverity, FusionDecision]:
        nlp_value = normalize_severity(inputs.nlp_result.severity if inputs.nlp_result else None)
        visual = vision_issue(inputs.vision_result)
        cv_value = normalize_severity(visual.visual_severity if visual else None)
        if cv_value and visual and visual.visual_confidence >= self.severity_threshold and nlp_value:
            if cv_value == nlp_value:
                return FusedSeverity(value=cv_value, source="fused", confidence=visual.visual_confidence, reason="Both modalities agree on normalized severity."), FusionDecision(field="severity", selected_source="fused", considered_sources=["vision", "nlp"], reason="Both modalities agree on normalized severity.")
            selected = cv_value if SEVERITY_RANK[cv_value] >= SEVERITY_RANK[nlp_value] else nlp_value
            selected_source = "vision" if selected == cv_value else "nlp"
            return FusedSeverity(value=selected, source=selected_source, confidence=visual.visual_confidence if selected_source == "vision" else inputs.nlp_result.confidence, reason="The higher severity was selected from reliable available evidence."), FusionDecision(field="severity", selected_source=selected_source, considered_sources=["vision", "nlp"], conflict=True, reason="The higher severity was selected from reliable available evidence.")
        if cv_value and visual and visual.visual_confidence >= self.severity_threshold:
            return FusedSeverity(value=cv_value, source="vision", confidence=visual.visual_confidence, reason="Vision supplied high-confidence severity."), FusionDecision(field="severity", selected_source="vision", considered_sources=["vision"], reason="Vision supplied high-confidence severity.")
        if nlp_value:
            return FusedSeverity(value=nlp_value, source="nlp", confidence=inputs.nlp_result.confidence, reason="NLP supplied the usable severity evidence."), FusionDecision(field="severity", selected_source="nlp", considered_sources=["vision", "nlp"], conflict=bool(cv_value and cv_value != nlp_value), reason="NLP was preferred because visual severity was unavailable or below threshold.")
        if cv_value:
            return FusedSeverity(value=cv_value, source="vision", confidence=visual.visual_confidence, reason="Vision supplied the only severity evidence."), FusionDecision(field="severity", selected_source="vision", considered_sources=["vision"], reason="Vision supplied the only severity evidence.")
        if visual and (visual.hazard_indicators or visual.obstruction_indicators) and visual.visible_scale in {"widespread", "large"}:
            return FusedSeverity(value="HIGH", source="rule", reason="Severity was escalated from explicit widespread hazard and obstruction evidence."), FusionDecision(field="severity", selected_source="rule", considered_sources=["vision"], reason="Widespread visual hazards and obstructions justify deterministic escalation.")
        return FusedSeverity(reason="No explicit severity evidence was available."), FusionDecision(field="severity", reason="No explicit severity evidence was available.")

    def resolve_location(self, inputs: FusionInput) -> tuple[FusedLocation | None, LocationContext, FusionDecision]:
        nlp = inputs.nlp_result
        context = LocationContext(nlp_location=nlp.location if nlp else None, nlp_landmark=nlp.landmark if nlp else None, geo_authoritative=inputs.geo_result is not None)
        if inputs.geo_result:
            geo = inputs.geo_result
            location_data = geo.model_dump()
            location_data["confidence"] = 1.0
            return FusedLocation(**location_data), context, FusionDecision(field="location", selected_source="geo", considered_sources=["geo", "nlp"], reason="Validated Geo coordinates are authoritative.")
        return None, context, FusionDecision(field="location", selected_source="nlp" if nlp and nlp.location else None, considered_sources=["geo", "nlp"], reason="NLP location remains textual context; coordinates were not fabricated.")

    def conflicts(self, inputs: FusionInput, category: FusedCategory, severity: FusedSeverity) -> list[FusionConflict]:
        nlp = inputs.nlp_result
        visual = vision_issue(inputs.vision_result)
        result = []
        if nlp and visual and nlp.problem_type and visual.visual_issue_type and normalize_term(nlp.problem_type) != normalize_term(visual.visual_issue_type):
            result.append(FusionConflict(field="issue_type", nlp_value=nlp.problem_type, vision_value=visual.visual_issue_type, selected_value=category.issue_type, selected_source=category.source, resolution=f"{category.source}_priority", reason=category.reason))
        if nlp and visual and nlp.primary_category and visual.visual_domain and normalize_term(nlp.primary_category) != normalize_term(visual.visual_domain):
            result.append(FusionConflict(field="category", nlp_value=nlp.primary_category, vision_value=visual.visual_domain, selected_value=category.value, selected_source=category.source, resolution=f"{category.source}_priority", reason=category.reason))
        nlp_severity = normalize_severity(nlp.severity if nlp else None)
        cv_severity = normalize_severity(visual.visual_severity if visual else None)
        if nlp_severity and cv_severity and nlp_severity != cv_severity:
            result.append(FusionConflict(field="severity", nlp_value=nlp_severity, vision_value=cv_severity, selected_value=severity.value, selected_source=severity.source, resolution=f"{severity.source}_priority", reason=severity.reason))
        if inputs.geo_result and nlp and nlp.location:
            location_text = " ".join(filter(None, [inputs.geo_result.district, inputs.geo_result.state])).lower()
            if location_text and nlp.location.lower() not in location_text and location_text not in nlp.location.lower():
                result.append(FusionConflict(field="location", nlp_value=nlp.location, geo_value=inputs.geo_result.district or inputs.geo_result.formatted_address, selected_value=inputs.geo_result.district, selected_source="geo", resolution="geo_priority", reason="Resolved Geo location is authoritative."))
        return result

    def build_priority(self, severity: FusedSeverity, inputs: FusionInput, visual: VisualIssue | None) -> FusedPriority:
        factors = []
        if severity.value in {"HIGH", "CRITICAL"}:
            factors.append(f"final severity is {severity.value}")
        if visual and visual.hazard_indicators:
            factors.append("visual hazards detected")
        if visual and visual.obstruction_indicators:
            factors.append("visual obstruction detected")
        if inputs.nlp_result and inputs.nlp_result.urgency:
            factors.append(f"NLP urgency: {inputs.nlp_result.urgency}")
        if inputs.nlp_result and inputs.nlp_result.affected_people:
            factors.append("affected people identified")
        if visual and visual.visible_scale and visual.visible_scale != "unknown":
            factors.append(f"visible scale: {visual.visible_scale}")
        if visual and visual.estimated_public_impact:
            factors.append("public impact described by vision")
        value = "HIGH" if severity.value == "CRITICAL" or len(factors) >= 2 else "MEDIUM" if severity.value == "HIGH" or factors else "LOW" if severity.value else None
        return FusedPriority(value=value, reason="Priority is derived from final fused severity and available impact factors.", factors=factors)

    def build(self, inputs: FusionInput) -> FusionResult:
        states = self.modality_states(inputs)
        category, category_decision = self.resolve_category(inputs)
        severity, severity_decision = self.resolve_severity(inputs)
        location, location_context, location_decision = self.resolve_location(inputs)
        visual = vision_issue(inputs.vision_result)
        nlp = inputs.nlp_result
        category_conflict = bool(category_decision.conflict)
        conflicts = self.conflicts(inputs, category, severity)
        priority = self.build_priority(severity, inputs, visual)
        department_value = DEPARTMENT_BY_CATEGORY.get(category.value or "")
        department = FusedDepartment(value=department_value, reason="Mapped from the final fused category." if department_value else "No authoritative department mapping is available for the final category.")
        routing = FusedRouting(department=department, reason=department.reason)
        action = visual.recommended_visual_action if visual and category.source == "vision" else nlp.suggested_action if nlp else None
        action_source = "vision" if visual and category.source == "vision" else "nlp" if nlp else None
        affected_people = nlp.affected_people if nlp and isinstance(nlp.affected_people, list) else []
        affected_objects = visual.affected_objects if visual else []
        evidence = Evidence(text=unique([nlp.complaint] if nlp else []), vision=unique(visual.visual_evidence if visual else []), geo=[f"Coordinates resolved to {inputs.geo_result.district or inputs.geo_result.state or 'an unresolved area'}." ] if inputs.geo_result else [])
        duplicate_features = DuplicateFeatures(category=category.value, issue_type=category.issue_type, domain=category.domain, normalized_complaint=normalize_term(nlp.complaint if nlp else None), visual_description=visual.visual_description if visual else None, relevant_evidence=unique(evidence.text + evidence.vision), coordinates={"lat": location.lat, "lng": location.lng} if location else None, district=location.district if location else None, block=location.block if location else None, state=location.state if location else None, severity=severity.value, landmark=nlp.landmark if nlp else None, affected_objects=affected_objects)
        availability = ModalityAvailability(nlp=states.nlp.usable, vision=states.vision.usable, geo=states.geo.usable)
        model_confidences = [value for value in [category.confidence, severity.confidence] if value is not None]
        fusion_status = "complete" if any((states.nlp.usable, states.vision.usable, states.geo.usable)) else "unresolved"
        return FusionResult(fusion_status=fusion_status, category=category, severity=severity, priority_candidate=priority.value, priority=priority, department=department, routing=routing, location=location, location_context=location_context, impact=ImpactSummary(affected_people=unique(affected_people), affected_objects=unique(affected_objects), public_impact=visual.estimated_public_impact if visual else None, scale=visual.visible_scale if visual else None), hazards=unique(visual.hazard_indicators if visual else []), environmental_indicators=unique(visual.environmental_indicators if visual else []), obstruction_indicators=unique(visual.obstruction_indicators if visual else []), visible_conditions=unique(visual.visible_conditions if visual else []), recommended_action={"action": action, "source": action_source}, evidence=evidence, modality_availability=availability, modality_states=states, confidence=ConfidenceSummary(overall=max(model_confidences) if model_confidences else None, category=category.confidence, severity=severity.confidence, location=location.confidence if location else None), conflicts=conflicts, decisions=[category_decision, severity_decision, location_decision], explainability=ExplainabilityMetadata(vision_category_confidence_threshold=self.category_threshold, vision_severity_confidence_threshold=self.severity_threshold, rules=["vision_category_threshold", "geo_authority", "highest_reliable_severity", "priority_from_fused_evidence"]), duplicate_features=duplicate_features)


fusion_service = FusionService()
