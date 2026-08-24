from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.geo import (
    ClusterRequest,
    ClusterResponse,
    LocationInput,
    ResolvedLocation,
)
from app.services.geo_service import GeoServiceError, geo_service

router = APIRouter(prefix="/api/v1/geo", tags=["Geo"])


@router.post("/resolve", response_model=ResolvedLocation)
async def resolve_location(payload: LocationInput) -> ResolvedLocation:
    """
    Accepts EITHER gps_coordinates OR manual_address, returns the single
    ResolvedLocation JSON used across NLP + vision + submission pipeline.
    """
    try:
        if payload.gps_coordinates:
            return await geo_service.reverse_geocode(
                payload.gps_coordinates.lat, payload.gps_coordinates.lng
            )
        return await geo_service.forward_geocode(payload.manual_address.text)
    except GeoServiceError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/cluster", response_model=ClusterResponse)
async def cluster_submissions(payload: ClusterRequest) -> ClusterResponse:
    """
    Groups nearby submissions (same category, within eps_meters) for
    duplicate/near-duplicate detection and map display.
    """
    return geo_service.cluster_points(
        points=[point.model_dump() for point in payload.points],
        eps_meters=payload.eps_meters,
        min_samples=payload.min_samples,
        group_by_category=payload.group_by_category,
    )
