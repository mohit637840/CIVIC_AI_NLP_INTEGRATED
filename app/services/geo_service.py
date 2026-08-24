from __future__ import annotations

import logging
from typing import Any, Iterable

import httpx
from sklearn.cluster import DBSCAN

from app.config import settings


logger = logging.getLogger(__name__)
NOMINATIM_URL = "https://nominatim.openstreetmap.org"


class GeoServiceError(Exception):
    """Controlled geospatial service error."""


class GeoService:
    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": getattr(
                settings,
                "NOMINATIM_USER_AGENT",
                "CIVIC_AI_NLP_INTEGRATED/0.1.0 (contact: civic-ai@example.com)",
            ),
            "Accept": "application/json",
        }

    @staticmethod
    def _validate_coordinates(lat: float, lng: float) -> None:
        if not -90 <= lat <= 90:
            raise GeoServiceError("Latitude must be between -90 and 90.")
        if not -180 <= lng <= 180:
            raise GeoServiceError("Longitude must be between -180 and 180.")

    @staticmethod
    def _is_missing(value: Any) -> bool:
        return value is None or (isinstance(value, str) and not value.strip())

    @staticmethod
    def _address_value(address: dict[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = address.get(key)
            if value is not None and (not isinstance(value, str) or value.strip()):
                return str(value).strip()
        return None

    @staticmethod
    def _resolution_status(district: str | None, block: str | None, state: str | None, formatted_address: str | None) -> str:
        present = [district, block, state, formatted_address]
        if not any(value is not None for value in present):
            return "coordinates_only"
        if all(value is not None for value in [district, state, formatted_address]) and block is None:
            return "partial"
        if all(value is not None for value in [district, block, state, formatted_address]):
            return "resolved"
        return "partial"

    async def reverse_geocode(self, lat: float, lng: float) -> dict[str, Any]:
        self._validate_coordinates(lat, lng)
        raw_input = f"{lat}, {lng}"

        params = {
            "lat": lat,
            "lon": lng,
            "format": "json",
            "addressdetails": 1,
        }

        try:
            async with httpx.AsyncClient(
                timeout=10.0,
                follow_redirects=True,
                headers=self._headers(),
            ) as client:
                response = await client.get(
                    f"{NOMINATIM_URL}/reverse",
                    params=params,
                )
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Reverse geocoding failed for %s, %s: %s", lat, lng, exc)
            return self._fallback_location(lat, lng, raw_input)

        if not isinstance(data, dict):
            logger.warning("Reverse geocoding returned an unexpected payload for %s, %s: %r", lat, lng, data)
            return self._fallback_location(lat, lng, raw_input)

        return self._build_location(lat, lng, raw_input, data)

    async def forward_geocode(self, address: str) -> dict[str, Any]:
        address = address.strip()
        if not address:
            raise GeoServiceError("Manual address cannot be empty.")

        params = {
            "q": address,
            "format": "json",
            "addressdetails": 1,
            "limit": 1,
        }

        try:
            async with httpx.AsyncClient(
                timeout=10.0,
                follow_redirects=True,
                headers=self._headers(),
            ) as client:
                response = await client.get(
                    f"{NOMINATIM_URL}/search",
                    params=params,
                )
                response.raise_for_status()
                results = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise GeoServiceError(
                f"Unable to resolve manual address: {exc}"
            ) from exc

        if not results:
            raise GeoServiceError(
                f"Could not resolve manual address: {address}"
            )

        item = results[0]
        try:
            lat = float(item["lat"])
            lng = float(item["lon"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GeoServiceError(
                "Geocoder returned invalid coordinates."
            ) from exc

        self._validate_coordinates(lat, lng)
        return self._build_location(
            lat,
            lng,
            address,
            item,
            source="manual",
        )

    @staticmethod
    def _build_location(
        lat: float,
        lng: float,
        raw_input: str,
        data: dict[str, Any],
        source: str = "gps",
    ) -> dict[str, Any]:
        address = data.get("address") or {}
        district = GeoService._address_value(
            address,
            "state_district",
            "district",
            "county",
            "city_district",
        )
        block = GeoService._address_value(
            address,
            "block",
            "municipality",
            "city",
            "town",
            "village",
            "suburb",
        )
        state = GeoService._address_value(address, "state")
        formatted_address = data.get("display_name")
        if isinstance(formatted_address, str):
            formatted_address = formatted_address.strip() or None

        resolution_status = GeoService._resolution_status(
            district=district,
            block=block,
            state=state,
            formatted_address=formatted_address,
        )

        return {
            "lat": lat,
            "lng": lng,
            "source": source,
            "raw_input": raw_input,
            "district": district,
            "block": block,
            "state": state,
            "formatted_address": formatted_address,
            "resolution_status": resolution_status,
        }

    @staticmethod
    def _fallback_location(
        lat: float,
        lng: float,
        raw_input: str,
    ) -> dict[str, Any]:
        return {
            "lat": lat,
            "lng": lng,
            "source": "gps",
            "raw_input": raw_input,
            "district": None,
            "block": None,
            "state": None,
            "formatted_address": None,
            "resolution_status": "coordinates_only",
        }

    @staticmethod
    def cluster_points(
        points: Iterable[dict[str, Any]],
        eps_km: float = 0.5,
        min_samples: int = 2,
        eps_meters: float | None = None,
        group_by_category: bool = True,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        points = list(points)
        if not points:
            return {"clusters": [], "noise_point_ids": []} if eps_meters is not None else []

        if eps_meters is not None:
            if eps_meters <= 0:
                raise GeoServiceError("eps_meters must be greater than zero.")
            eps_km = eps_meters / 1000
        elif eps_km <= 0:
            raise GeoServiceError("eps_km must be greater than zero.")
        if min_samples < 1:
            raise GeoServiceError("min_samples must be at least 1.")

        def point_coordinates(point: dict[str, Any]) -> tuple[float, float]:
            location = point.get("location", point)
            return float(location["lat"]), float(location["lng"])

        coords = [list(point_coordinates(point)) for point in points]

        # Approximate lat/lng distance in kilometres using a local
        # equirectangular projection. Good enough for hotspot clustering.
        import math

        lat0 = math.radians(sum(p[0] for p in coords) / len(coords))
        earth_km = 6371.0088
        projected = [
            [
                earth_km * math.radians(lat - coords[0][0]),
                earth_km * math.cos(lat0) * math.radians(lng - coords[0][1]),
            ]
            for lat, lng in coords
        ]

        labels = DBSCAN(eps=eps_km, min_samples=min_samples).fit_predict(projected)

        if eps_meters is not None and group_by_category:
            labels = labels.copy()
            next_label = 0
            for category in sorted({point.get("category") for point in points}):
                category_indices = [
                    index for index, point in enumerate(points)
                    if point.get("category") == category
                ]
                category_labels = DBSCAN(
                    eps=eps_km,
                    min_samples=min_samples,
                ).fit_predict([projected[index] for index in category_indices])
                for index, category_label in zip(category_indices, category_labels):
                    labels[index] = (
                        next_label + int(category_label)
                        if category_label >= 0
                        else -1
                    )
                if any(category_labels >= 0):
                    next_label += int(max(category_labels)) + 1

        groups: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            if label >= 0:
                groups.setdefault(int(label), []).append(idx)

        result = []
        for label, indices in sorted(groups.items()):
            center_lat = sum(coords[i][0] for i in indices) / len(indices)
            center_lng = sum(coords[i][1] for i in indices) / len(indices)
            if eps_meters is None:
                result.append({
                    "cluster_id": label,
                    "count": len(indices),
                    "points": [points[i] for i in indices],
                    "centroid": {"lat": center_lat, "lng": center_lng},
                })
            else:
                result.append({
                    "cluster_id": label,
                    "category": points[indices[0]].get("category"),
                    "point_ids": [points[i]["id"] for i in indices],
                    "center_lat": center_lat,
                    "center_lng": center_lng,
                    "count": len(indices),
                })

        if eps_meters is not None:
            clustered = {index for indices in groups.values() for index in indices}
            return {
                "clusters": result,
                "noise_point_ids": [
                    point["id"] for index, point in enumerate(points)
                    if index not in clustered
                ],
            }

        return result


geo_service = GeoService()
