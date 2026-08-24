from __future__ import annotations

from app.schemas.submission import SubmissionContext


class RoutingService:
    """Expose routing derived from the final fused category."""

    def process(self, submission: SubmissionContext) -> SubmissionContext:
        submission.routing = submission.fusion.routing
        return submission


routing_service = RoutingService()