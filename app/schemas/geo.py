from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class LocationSource(str, Enum):
    GPS = "gps"
    MANUAL = "manual"


class GPSCoordinates(BaseModel):
    """Raw GPS coordinates sent by the device."""

    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)


class ManualAddress(BaseModel):
    """Free-text location typed in by the user."""

    text: str = Field(..., min_length=3, description="e.g. 'Ratu, Ranchi'")


class LocationInput(BaseModel):
    """
    What the CLIENT sends. Exactly one of the two fields is filled —
    never both, never neither.
    """

    gps_coordinates: Optional[GPSCoordinates] = None
    manual_address: Optional[ManualAddress] = None

    @model_validator(mode="after")
    def check_exactly_one_source(self) -> "LocationInput":
        if bool(self.gps_coordinates) == bool(self.manual_address):
            raise ValueError(
                "Provide exactly one of 'gps_coordinates' or 'manual_address', not both/neither."
            )
        return self


class ResolvedLocation(BaseModel):
    """
    What gets RETURNED/STORED after geocoding — always one single object,
    regardless of which input path was used. 'source' records which path.
    """

    lat: float
    lng: float
    source: LocationSource
    raw_input: Optional[str] = None
    district: Optional[str] = None
    block: Optional[str] = None
    state: Optional[str] = None
    formatted_address: Optional[str] = None
    resolution_status: str = "resolved"


class SubmissionPoint(BaseModel):
    """Minimal shape needed to place + cluster a submission on the map."""

    id: str
    category: str
    severity: Optional[str] = None
    summary: Optional[str] = None
    location: ResolvedLocation


class ClusterRequest(BaseModel):
    points: list[SubmissionPoint]
    eps_meters: float = Field(500.0, description="Max distance between points in a cluster")
    min_samples: int = Field(2, ge=1)
    group_by_category: bool = True


class ClusterGroup(BaseModel):
    cluster_id: int
    category: Optional[str] = None
    point_ids: list[str]
    center_lat: float
    center_lng: float
    count: int


class ClusterResponse(BaseModel):
    clusters: list[ClusterGroup]
    noise_point_ids: list[str] = Field(
        default_factory=list, description="Points that didn't join any cluster"
    )
