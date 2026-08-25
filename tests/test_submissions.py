from fastapi.testclient import TestClient

from app.main import app
from app.schemas.submission import SubmissionContext
from app.schemas.vision import VisionAnalysisResult, VisualIssue
from app.services.duplicate_service import duplicate_service
from app.services.priority_service import priority_service
from app.services.routing_service import routing_service


client = TestClient(app)


def test_downstream_processors_consume_canonical_submission_context():
    response = client.post(
        "/api/v1/submissions",
        json={"description": "There is a pothole near the school"},
    )

    context = SubmissionContext(**response.json())
    duplicate_context = duplicate_service.process(context)
    priority_context = priority_service.process(duplicate_context)
    routing_context = routing_service.process(priority_context)

    assert routing_context is context
    assert routing_context.fusion.category.value
    assert routing_context.priority == routing_context.fusion.priority
    assert routing_context.routing == routing_context.fusion.routing
    assert routing_context.duplicate is not None


def test_submission_text_only():
    response = client.post(
        "/api/v1/submissions",
        json={"description": "There is a pothole on the road near the school"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["description"]["available"] is True
    assert body["description"]["analysis"]["complaint"]
    assert body["image"]["available"] is False
    assert body["location"] is None
    assert body["fusion"]["fusion_status"] == "complete"
    assert body["fusion"]["category"]["source"] == "nlp"


def test_submission_text_and_gps(monkeypatch):
    async def fake_reverse_geocode(lat, lng):
        return {"lat": lat, "lng": lng, "source": "gps", "raw_input": f"{lat}, {lng}"}

    monkeypatch.setattr(
        "app.services.submission_service.geo_service.reverse_geocode",
        fake_reverse_geocode,
    )
    response = client.post(
        "/api/v1/submissions",
        json={
            "description": "There is severe waterlogging near Main Market in Ranchi.",
            "location": {"gps_coordinates": {"lat": 23.3441, "lng": 85.3096}},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["description"]["available"] is True
    assert body["location"]["lat"] == 23.3441
    assert body["image"]["available"] is False
    assert body["fusion"]["location"]["source"] == "gps"


def test_submission_text_image_and_gps_preserves_all_modalities(monkeypatch):
    async def fake_reverse_geocode(lat, lng):
        return {"lat": lat, "lng": lng, "source": "gps", "district": "Ranchi"}

    async def fake_vision(*, image_bytes, mime_type, user_description):
        return VisionAnalysisResult(image_valid=True, issue_detected=False)

    monkeypatch.setattr(
        "app.services.submission_service.geo_service.reverse_geocode",
        fake_reverse_geocode,
    )
    monkeypatch.setattr("app.services.submission_service.vision_service.analyze", fake_vision)
    response = client.post(
        "/api/v1/submissions",
        json={
            "description": "There is waterlogging near Main Market in Ranchi.",
            "image_base64": "aW1hZ2U=",
            "location": {"gps_coordinates": {"lat": 23.3441, "lng": 85.3096}},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["description"]["available"] is True
    assert body["image"]["available"] is True
    assert body["image"]["analysis"]["image_valid"] is True
    assert body["location"]["district"] == "Ranchi"
    assert body["fusion"]["modality_availability"] == {
        "nlp": True,
        "vision": False,
        "geo": True,
    }


def test_submission_conflict_preserves_raw_outputs_and_uses_vision(monkeypatch):
    async def fake_reverse_geocode(lat, lng):
        return {
            "lat": lat,
            "lng": lng,
            "source": "gps",
            "district": "Ranchi",
            "state": "Jharkhand",
        }

    async def fake_vision(*, image_bytes, mime_type, user_description):
        return VisionAnalysisResult(
            image_valid=True,
            issue_detected=True,
            issues=[VisualIssue(
                visual_domain="Water / Drainage",
                visual_issue_type="Waterlogging / Flooding",
                visual_severity="high",
                visual_confidence=0.95,
                visual_description="The road is covered by standing water.",
                visual_evidence=["Standing water covers the road"],
                affected_objects=["Road infrastructure"],
                hazard_indicators=["Vehicle damage risk"],
                environmental_indicators=["Standing water"],
                obstruction_indicators=["Road obstruction"],
                visible_conditions=["Deep standing water"],
                estimated_public_impact="Vehicles are having difficulty passing.",
                recommended_visual_action="Inspect drainage and remove blockage.",
            )],
        )

    monkeypatch.setattr(
        "app.services.submission_service.geo_service.reverse_geocode",
        fake_reverse_geocode,
    )
    monkeypatch.setattr("app.services.submission_service.vision_service.analyze", fake_vision)
    response = client.post(
        "/api/v1/submissions",
        json={
            "description": "There is a huge pothole on the road near Main Market in Ranchi.",
            "image_base64": "aW1hZ2U=",
            "location": {"gps_coordinates": {"lat": 23.3441, "lng": 85.3096}},
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["description"]["analysis"]["problem_type"] == "Road Damage / Potholes"
    assert body["image"]["analysis"]["issues"][0]["visual_issue_type"] == "Waterlogging / Flooding"
    assert body["fusion"]["category"]["source"] == "vision"
    assert body["fusion"]["category"]["issue_type"] == "Waterlogging / Flooding"
    assert body["fusion"]["severity"]["source"] == "vision"
    assert body["fusion"]["location"]["source"] == "gps"
    assert {conflict["field"] for conflict in body["fusion"]["conflicts"]} >= {"category", "issue_type", "location"}
    assert body["priority"]["value"] == "HIGH"
    assert body["routing"]["department"]["value"] == "Water Supply Department"


def test_submission_nlp_failure_does_not_block_cv_and_geo(monkeypatch):
    async def fake_reverse_geocode(lat, lng):
        return {"lat": lat, "lng": lng, "source": "gps"}

    def failing_nlp(text):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("app.services.submission_service.nlp_service.analyze", failing_nlp)
    monkeypatch.setattr(
        "app.services.submission_service.geo_service.reverse_geocode",
        fake_reverse_geocode,
    )
    response = client.post(
        "/api/v1/submissions",
        json={
            "description": "A civic issue",
            "location": {"gps_coordinates": {"lat": 23.3441, "lng": 85.3096}},
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["description"]["available"] is False
    assert body["fusion"]["modality_states"]["nlp"]["status"] == "failed"
    assert body["fusion"]["location"]["source"] == "gps"


def test_submission_geo_failure_does_not_block_nlp(monkeypatch):
    async def failing_reverse_geocode(lat, lng):
        raise RuntimeError("geocoder unavailable")

    monkeypatch.setattr(
        "app.services.submission_service.geo_service.reverse_geocode",
        failing_reverse_geocode,
    )
    response = client.post(
        "/api/v1/submissions",
        json={
            "description": "There is a pothole near the school.",
            "location": {"gps_coordinates": {"lat": 23.3441, "lng": 85.3096}},
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["description"]["available"] is True
    assert body["location"] is None
    assert body["location_error"] == "Location resolution is currently unavailable"
    assert body["fusion"]["modality_states"]["geo"]["status"] == "failed"


def test_submission_image_only_reports_unavailable_cv_without_key():
    response = client.post("/api/v1/submissions", json={"image_base64": "aW1hZ2U="})

    assert response.status_code == 200
    body = response.json()
    assert body["description"]["available"] is False
    assert body["image"]["available"] is False
    assert body["image"]["error"]


def test_submission_rejects_empty_payload():
    response = client.post("/api/v1/submissions", json={})

    assert response.status_code == 422
    assert "at least one" in response.json()["detail"][0]["msg"]


def test_submission_openapi_has_input_and_output_models():
    document = app.openapi()
    schemas = document["components"]["schemas"]
    operation = document["paths"]["/api/v1/submissions"]["post"]

    request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    response_schema = operation["responses"]["200"]["content"]["application/json"]["schema"]

    assert request_schema["$ref"].endswith("SubmissionRequest")
    assert response_schema["$ref"].endswith("SubmissionResponse")
    assert schemas["SubmissionRequest"]["example"]["description"]
    assert schemas["SubmissionResponse"]["properties"]["description"]["$ref"].endswith(
        "DescriptionResult"
    )


def test_geo_cluster_endpoint_returns_declared_shape():
    payload = {
        "points": [
            {
                "id": "a",
                "category": "roads",
                "location": {"lat": 23.3441, "lng": 85.3096, "source": "gps"},
            },
            {
                "id": "b",
                "category": "roads",
                "location": {"lat": 23.3445, "lng": 85.3100, "source": "gps"},
            },
        ],
        "eps_meters": 500,
        "min_samples": 2,
    }

    response = client.post("/api/v1/geo/cluster", json=payload)

    assert response.status_code == 200
    assert response.json()["clusters"][0]["point_ids"] == ["a", "b"]