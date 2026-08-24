import asyncio

import httpx
import pytest

from app.services.geo_service import GeoService, GeoServiceError


def test_validate_coordinates():
    GeoService._validate_coordinates(23.3441, 85.3096)


def test_invalid_latitude():
    with pytest.raises(GeoServiceError):
        GeoService._validate_coordinates(91, 85.3096)


def test_cluster_points():
    points = [
        {"lat": 23.3441, "lng": 85.3096},
        {"lat": 23.3442, "lng": 85.3097},
        {"lat": 23.50, "lng": 85.60},
    ]
    clusters = GeoService.cluster_points(points, eps_km=1, min_samples=2)
    assert len(clusters) == 1
    assert clusters[0]["count"] == 2


def test_reverse_geocode_enriches_authoritative_fields():
    result = asyncio.run(GeoService().reverse_geocode(23.3441, 85.3096))

    assert result["source"] == "gps"
    assert result["lat"] == 23.3441
    assert result["lng"] == 85.3096
    assert result["district"] or result["state"]
    assert result["state"] == "Jharkhand"
    assert result["formatted_address"]
    assert result["resolution_status"] in {"resolved", "partial"}


def test_reverse_geocode_falls_back_to_coordinates_only_on_provider_failure(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            raise httpx.HTTPError("provider unavailable")

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    result = asyncio.run(GeoService().reverse_geocode(23.3441, 85.3096))

    assert result["district"] is None
    assert result["block"] is None
    assert result["state"] is None
    assert result["formatted_address"] is None
    assert result["resolution_status"] == "coordinates_only"


def test_reverse_geocode_partial_metadata_sets_partial_status():
    partial = {
        "display_name": "Ranchi, Jharkhand, India",
        "address": {"state": "Jharkhand"},
    }

    result = GeoService._build_location(23.3441, 85.3096, "23.3441, 85.3096", partial)

    assert result["district"] is None
    assert result["state"] == "Jharkhand"
    assert result["formatted_address"] == "Ranchi, Jharkhand, India"
    assert result["resolution_status"] == "partial"


def test_forward_geocode_manual_address_works(monkeypatch):
    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            return FakeResponse([
                {
                    "lat": "23.3441",
                    "lon": "85.3096",
                    "display_name": "Main Market, Ranchi, Jharkhand, India",
                    "address": {"state_district": "Ranchi", "state": "Jharkhand"},
                }
            ])

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    result = asyncio.run(GeoService().forward_geocode("Main Market, Ranchi"))

    assert result["source"] == "manual"
    assert result["district"] == "Ranchi"
    assert result["state"] == "Jharkhand"
    assert result["formatted_address"]
    assert result["resolution_status"] in {"resolved", "partial"}
