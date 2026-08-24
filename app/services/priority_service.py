from __future__ import annotations

from app.schemas.submission import SubmissionContext


class PriorityService:
    """Expose priority derived from the final fused evidence."""

    def process(self, submission: SubmissionContext) -> SubmissionContext:
        submission.priority = submission.fusion.priority
        return submission


priority_service = PriorityService()