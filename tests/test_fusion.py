from app.schemas.fusion import FusionInput
from app.schemas.geo import ResolvedLocation
from app.schemas.nlp import NLPAnalysis
from app.schemas.vision import VisionAnalysisResult, VisualIssue
from app.services.fusion_service import FusionService


def nlp(**overrides):
    values = {
        "complaint": "There is waterlogging near Main Market in Ranchi.",
        "primary_category": "Road & Urban Infrastructure",
        "problem_type": "Road Damage / Potholes",
        "domain": "Urban Infrastructure",
        "severity": "Medium",
        "location": "Jamshedpur",
        "landmark": "Main Market",
        "affected_people": ["Residents"],
        "suggested_action": "Inspect the road",
    }
    values.update(overrides)
    return NLPAnalysis(**values)


def vision(confidence=0.95, issue_type="Waterlogging / Flooding", severity="high"):
    return VisionAnalysisResult(
        image_valid=True,
        issue_detected=True,
        issues=[VisualIssue(
            visual_domain="Water / Drainage",
            visual_issue_type=issue_type,
            visual_severity=severity,
            visual_confidence=confidence,
            visual_description="Standing water covers the road.",
            visual_evidence=["Road covered by standing water"],
            affected_objects=["Road infrastructure"],
            hazard_indicators=["Vehicle damage risk"],
            environmental_indicators=["Standing water"],
            obstruction_indicators=["Road obstruction"],
            visible_conditions=["Deep water"],
            estimated_public_impact="Traffic is disrupted.",
            recommended_visual_action="Inspect drainage and remove blockage.",
        )],
    )


def geo(district="Ranchi"):
    return ResolvedLocation(
        lat=23.3441,
        lng=85.3096,
        source="gps",
        district=district,
        state="Jharkhand",
    )


def test_fusion_supports_all_modality_combinations():
    service = FusionService()
    combinations = [
        FusionInput(nlp_result=nlp(), vision_result=vision(), geo_result=geo()),
        FusionInput(nlp_result=nlp(), geo_result=geo()),
        FusionInput(vision_result=vision(), geo_result=geo()),
        FusionInput(nlp_result=nlp(), vision_result=vision()),
        FusionInput(nlp_result=nlp()),
        FusionInput(vision_result=vision()),
        FusionInput(geo_result=geo()),
        FusionInput(),
    ]

    results = [service.build(item) for item in combinations]

    assert all(result is not None for result in results)
    assert results[0].fusion_status == "complete"
    assert results[-1].fusion_status == "unresolved"


def test_high_confidence_vision_wins_category_and_records_conflict():
    result = FusionService().build(FusionInput(nlp_result=nlp(), vision_result=vision()))

    assert result.category.source == "vision"
    assert result.category.value == "Water / Drainage"
    assert result.conflicts[0].field == "issue_type"
    assert result.conflicts[0].resolution == "vision_priority"


def test_low_confidence_vision_falls_back_to_nlp():
    result = FusionService().build(FusionInput(nlp_result=nlp(), vision_result=vision(0.4)))

    assert result.category.source == "nlp"
    assert result.category.value == "Road & Urban Infrastructure"
    assert result.conflicts[0].resolution == "nlp_priority"


def test_geo_wins_location_and_records_conflict():
    result = FusionService().build(FusionInput(nlp_result=nlp(), geo_result=geo()))

    assert result.location.source == "gps"
    assert result.location_context.geo_authoritative is True
    assert any(conflict.field == "location" for conflict in result.conflicts)


def test_severity_uses_highest_normalized_signal():
    result = FusionService().build(
        FusionInput(
            nlp_result=nlp(severity="medium"),
            vision_result=vision(severity="high"),
        )
    )

    assert result.severity.value == "HIGH"
    assert result.severity.source == "vision"
    assert any(conflict.field == "severity" for conflict in result.conflicts)


def test_agreement_has_no_issue_conflict_and_missing_confidence_stays_null():
    result = FusionService().build(
        FusionInput(
            nlp_result=nlp(problem_type="Waterlogging / Flooding", confidence=None),
            vision_result=vision(issue_type="Waterlogging / Flooding", confidence=0.8),
        )
    )

    assert any(conflict.field == "issue_type" for conflict in result.conflicts) is False
    assert result.category.confidence == 0.8
    assert result.severity.confidence == 0.8


def test_agreeing_severity_is_fused():
    result = FusionService().build(
        FusionInput(
            nlp_result=nlp(severity="High"),
            vision_result=vision(severity="high"),
        )
    )

    assert result.severity.value == "HIGH"
    assert result.severity.source == "fused"


def test_priority_routing_and_duplicate_features_use_final_category():
    result = FusionService().build(
        FusionInput(nlp_result=nlp(), vision_result=vision(), geo_result=geo())
    )

    assert result.priority.value == "HIGH"
    assert result.priority_candidate == "HIGH"
    assert result.routing.department.value == "Water Supply Department"
    assert result.duplicate_features.category == "Water / Drainage"
    assert result.duplicate_features.coordinates == {"lat": 23.3441, "lng": 85.3096}


def test_gps_coordinates_only_location_is_preserved():
    result = FusionService().build(
        FusionInput(geo_result=geo())
    )

    assert result.location.resolution_status == "resolved"
    assert result.location.lat == 23.3441


def test_failed_modality_is_explicitly_marked():
    result = FusionService().build(FusionInput(nlp_error="model unavailable"))

    assert result.fusion_status == "unresolved"
    assert result.modality_states.nlp.status == "failed"
    assert result.modality_states.nlp.error == "model unavailable"