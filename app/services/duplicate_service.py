from __future__ import annotations

from app.schemas.submission import SubmissionContext


class DuplicateService:
    """Future duplicate processor that does not fabricate a result."""

    def process(self, submission: SubmissionContext) -> SubmissionContext:
        return submission


duplicate_service = DuplicateService()