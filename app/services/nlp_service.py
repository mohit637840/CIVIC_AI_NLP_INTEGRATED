from __future__ import annotations

from typing import Any


class NLPService:
    """Adapter around the teammate's existing trained NLP pipeline.

    The underlying TF-IDF + classifier + rule-extraction implementation is kept
    intact. This service only exposes a stable application-level callable.
    """

    def __init__(self) -> None:
        self._predict = None

    def _load(self):
        if self._predict is None:
            from app.integrations.nlp.teammate_nlp import predict_complaint

            self._predict = predict_complaint
        return self._predict

    def analyze(self, text: str) -> dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("Complaint text cannot be empty")
        return self._load()(text)


nlp_service = NLPService()
