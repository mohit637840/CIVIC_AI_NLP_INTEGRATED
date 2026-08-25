from app.schemas.fusion import FusionResult, FusedCategory, FusedSeverity, FusedPriority, FusedDepartment, FusedRouting, ImpactSummary, Evidence, ModalityAvailability, ModalityStates, ModalityState, ConfidenceSummary, LocationContext
from app.schemas.submission import DescriptionResult, SubmissionContext
from app.services.duplicate_service import ContradictionDetector, IssueNormalizer, duplicate_service


def _fusion(category="Water / Drainage", issue_type="Waterlogging / Flooding"):
    return FusionResult(
        fusion_status="complete",
        category=FusedCategory(
            value=category,
            issue_type=issue_type,
            domain="Water / Drainage",
            source="nlp",
            confidence=0.9,
            reason="Test",
        ),
        severity=FusedSeverity(value="HIGH", source="nlp", confidence=0.9, reason="Test"),
        priority_candidate="HIGH",
        priority=FusedPriority(value="HIGH", source="rule_engine", reason="Test", factors=["test"]),
        department=FusedDepartment(value="Water Supply Department", source="category_mapping", reason="Test"),
        routing=FusedRouting(department=FusedDepartment(value="Water Supply Department", source="category_mapping", reason="Test"), reason="Test"),
        location=None,
        location_context=LocationContext(nlp_location="Main Market, Ranchi", nlp_landmark="Main Market", geo_authoritative=False),
        impact=ImpactSummary(affected_people=["Residents"], affected_objects=["Road infrastructure"], public_impact="Traffic is disrupted.", scale="localized"),
        hazards=[],
        environmental_indicators=["Standing water"],
        obstruction_indicators=["Road obstruction"],
        visible_conditions=["Deep water"],
        recommended_action={"action": "Inspect drainage", "source": "vision"},
        evidence=Evidence(text=["There is severe waterlogging near Main Market in Ranchi."], vision=[], geo=[]),
        modality_availability=ModalityAvailability(nlp=True, vision=False, geo=False),
        modality_states=ModalityStates(
            nlp=ModalityState(available=True, usable=True, status="valid", error=None),
            vision=ModalityState(available=False, usable=False, status="not_provided", error=None),
            geo=ModalityState(available=False, usable=False, status="not_provided", error=None),
        ),
        confidence=ConfidenceSummary(overall=0.9, category=0.9, severity=0.9, location=None),
        explainability={
            "engine": "rule_based",
            "version": "1.0",
            "vision_category_confidence_threshold": 0.75,
            "vision_severity_confidence_threshold": 0.75,
            "rules": ["test"],
        },
        duplicate_features={
            "category": category,
            "issue_type": issue_type,
            "domain": "Water / Drainage",
            "normalized_complaint": "severe waterlogging near main market in ranchi.",
            "visual_description": "Standing water covers the road.",
            "relevant_evidence": ["There is severe waterlogging near Main Market in Ranchi."],
            "coordinates": None,
            "district": "Ranchi",
            "block": None,
            "state": "Jharkhand",
            "severity": "HIGH",
            "landmark": "Main Market",
            "affected_objects": ["Road infrastructure"],
        },
    )


def _submission(text: str, submission_id: str = "submission-1"):
    return SubmissionContext(
        id=submission_id,
        description=DescriptionResult(available=True, analysis={
            "complaint": text,
            "language": "English",
            "primary_category": "Water / Drainage",
            "problem_type": "Waterlogging / Flooding",
            "domain": "Water / Drainage",
            "severity": "HIGH",
            "location": "Main Market, Ranchi",
            "landmark": "Main Market",
        }),
        image={"available": False, "provider": None, "model": None, "analysis": None, "error": None},
        location=None,
        fusion=_fusion(),
        duplicate=None,
        priority=None,
        routing=None,
        status="pending_review",
    )


def test_duplicate_detects_repeated_waterlogging_issue():
    duplicate_service.reset()
    first = _submission("There is severe waterlogging near Main Market in Ranchi. Vehicles cannot pass.", "a")
    second = _submission("Main Market road is flooded with standing water and traffic is severely affected.", "b")

    duplicate_service.process(first)
    result = duplicate_service.process(second)

    assert result.duplicate is not None
    assert result.duplicate["status"] in {"duplicate", "possible_duplicate"}
    assert result.duplicate["best_match"]["submission_id"] == "a"
    assert len(result.duplicate["candidates"]) >= 1


def test_duplicate_excludes_self_match():
    duplicate_service.reset()
    submission = _submission("Severe waterlogging near Main Market.", "self")

    result = duplicate_service.process(submission)

    assert result.duplicate is not None
    assert result.duplicate["status"] == "no_candidates"


def test_duplicate_empty_repository_returns_no_candidates():
    duplicate_service.reset()
    submission = _submission("Severe waterlogging near Main Market.", "new")

    result = duplicate_service.process(submission)

    assert result.duplicate is not None
    assert result.duplicate["status"] == "no_candidates"


def test_duplicate_rejects_active_vs_resolved_state_conflict():
    duplicate_service.reset()
    first = _submission("Road is flooded near Main Market.", "a")
    second = _submission("Road is no longer flooded near Main Market.", "b")

    duplicate_service.process(first)
    result = duplicate_service.process(second)

    assert result.duplicate is not None
    assert result.duplicate["status"] == "not_duplicate"
    assert "STATE_CONFLICT" in result.duplicate["reason_codes"] or "ISSUE_RESOLVED" in result.duplicate["reason_codes"]


def test_duplicate_rejects_same_location_different_issue():
    duplicate_service.reset()
    first = _submission("Severe waterlogging near Main Market.", "a")
    second = _submission("Streetlight is broken near Main Market.", "b")

    duplicate_service.process(first)
    result = duplicate_service.process(second)

    assert result.duplicate is not None
    assert result.duplicate["status"] == "not_duplicate"
    assert "ISSUE_MISMATCH" in result.duplicate["reason_codes"]


def test_duplicate_rejects_same_category_different_issue():
    duplicate_service.reset()
    first = _submission("Pothole near Main Market.", "a")
    second = _submission("Waterlogging near Main Market.", "b")

    duplicate_service.process(first)
    result = duplicate_service.process(second)

    assert result.duplicate is not None
    assert result.duplicate["status"] == "not_duplicate"
    assert "ISSUE_MISMATCH" in result.duplicate["reason_codes"] or "CATEGORY_MISMATCH" in result.duplicate["reason_codes"]


def test_issue_normalizer_handles_english_hindi_and_hinglish_potholes():
    assert IssueNormalizer.canonicalize("There is a large potholes on the road") == "pothole"
    assert IssueNormalizer.canonicalize("सड़क में बड़ा गड्ढा है") == "pothole"
    assert IssueNormalizer.canonicalize("Road par bada pothole hai") == "pothole"


def test_issue_normalizer_handles_waterlogging_variants():
    assert IssueNormalizer.canonicalize("जलभराव और पानी जमा है") == "waterlogging"
    assert IssueNormalizer.canonicalize("The road is flooded") == "waterlogging"


def test_negation_is_contextual_for_failure_states():
    active = ContradictionDetector.extract_issue_state("The streetlight is not working.")
    negated = ContradictionDetector.extract_issue_state("There is no pothole on the road.")

    assert active["state"] == "active"
    assert active["negated"] is False
    assert negated["negated"] is True


def test_active_states_do_not_create_state_conflict():
    result = ContradictionDetector.evaluate(
        "The pothole is active.",
        "There is a pothole on the road.",
        "pothole",
        "pothole",
    )

    assert result["state_compatibility"] == 1.0
    assert result["explicit_state_contradiction"] is False
    assert "STATE_CONFLICT" not in result["reason_codes"]


def test_unknown_state_is_unavailable_not_a_conflict():
    result = ContradictionDetector.evaluate(
        "A civic concern was recorded.",
        "Pothole near the market.",
        "pothole",
        "pothole",
    )

    assert result["state_compatibility"] is None
    assert result["explicit_state_contradiction"] is False
    assert "STATE_CONFLICT" not in result["reason_codes"]


def test_resolution_is_distinct_from_active_state():
    resolved = ContradictionDetector.extract_issue_state("The pothole was repaired.")

    assert resolved["state"] == "resolved"
    assert resolved["negated"] is False


def test_duplicate_output_marks_visual_evidence_unavailable_without_cv():
    duplicate_service.reset()
    first = _submission("There is severe waterlogging near Main Market in Ranchi.", "a")
    second = _submission("Main Market road is flooded.", "b")

    duplicate_service.process(first)
    result = duplicate_service.process(second)

    assert result.duplicate["signals"]["visual_evidence_compatibility"] is None
    assert result.duplicate["signals"]["visual_similarity"] is None
    assert "MISSING_STRUCTURED_VISUAL_EVIDENCE" in result.duplicate["reason_codes"]
    assert result.duplicate["provenance"]["version"] == "1.2"
