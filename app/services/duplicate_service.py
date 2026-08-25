from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings
from app.schemas.submission import SubmissionContext


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    cleaned = re.sub(r"[^\w\s\u0900-\u097F]", " ", str(value).lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class EmbeddingProvider:
    """Local embedding provider used by the first duplicate detector."""

    def __init__(self) -> None:
        self.provider = settings.EMBEDDING_PROVIDER
        self.model = settings.EMBEDDING_MODEL
        self.version = settings.EMBEDDING_VERSION
        self.vectorizer: TfidfVectorizer | None = None

    def _build_vectorizer(self, texts: list[str]) -> TfidfVectorizer:
        cleaned = [normalize_text(text) for text in texts if normalize_text(text)]
        if not cleaned:
            return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, strip_accents="unicode")
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            strip_accents="unicode",
        ).fit(cleaned)

    def fit(self, texts: list[str]) -> None:
        self.vectorizer = self._build_vectorizer(texts)

    @property
    def dimension(self) -> int:
        if self.vectorizer is None:
            return 0
        return len(self.vectorizer.vocabulary_)

    def embed_text(self, text: str | None) -> np.ndarray | None:
        clean = normalize_text(text)
        if not clean:
            return None
        if self.vectorizer is None:
            self.vectorizer = self._build_vectorizer([clean])
        return self.vectorizer.transform([clean]).toarray()[0]

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        cleaned = [normalize_text(text) for text in texts if normalize_text(text)]
        if not cleaned:
            return []
        self.fit(cleaned)
        return [row for row in self.vectorizer.transform(cleaned).toarray()]


class EmbeddingRepository:
    """In-memory repository that hides storage implementation behind a simple interface."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, dict[str, Any]]] = {}
        self.provider = EmbeddingProvider()

    def reset(self) -> None:
        self._records.clear()

    def refresh_vectors(self) -> None:
        if not self._records:
            return
        payloads: list[str] = []
        for bucket in self._records.values():
            for item in bucket.values():
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    payloads.append(text)
        if not payloads:
            return
        self.provider.fit(list(dict.fromkeys(payloads)))
        for bucket in self._records.values():
            for item in bucket.values():
                text = item.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                vector = self.provider.embed_text(text)
                if vector is None:
                    continue
                item["vector"] = np.asarray(vector, dtype=float).reshape(-1)
                item["metadata"]["dimension"] = int(item["vector"].size)

    def save(
        self,
        submission_id: str,
        embedding_type: str,
        vector: np.ndarray,
        metadata: dict[str, Any],
        text: str | None = None,
    ) -> None:
        vector_array = np.asarray(vector, dtype=float).reshape(-1)
        self._records.setdefault(str(submission_id), {})[embedding_type] = {
            "submission_id": str(submission_id),
            "embedding_type": embedding_type,
            "vector": vector_array,
            "metadata": {
                "submission_id": str(submission_id),
                "embedding_type": embedding_type,
                "provider": metadata.get("provider", settings.EMBEDDING_PROVIDER),
                "model": metadata.get("model", settings.EMBEDDING_MODEL),
                "dimension": int(metadata.get("dimension") or vector_array.size),
                "version": metadata.get("version", settings.EMBEDDING_VERSION),
                "created_at": metadata.get("created_at") or datetime.now(timezone.utc).isoformat(),
            },
            "text": text,
        }

    def get(self, submission_id: str, embedding_type: str) -> dict[str, Any] | None:
        return self._records.get(str(submission_id), {}).get(embedding_type)

    def search(self, vector: np.ndarray, top_k: int, embedding_type: str) -> list[dict[str, Any]]:
        if not self._records:
            return []
        query = np.asarray(vector, dtype=float).reshape(-1)
        matches: list[dict[str, Any]] = []
        for submission_id, bucket in self._records.items():
            item = bucket.get(embedding_type)
            if item is None:
                continue
            candidate_vector = np.asarray(item["vector"], dtype=float).reshape(-1)
            if candidate_vector.size != query.size:
                continue
            similarity = float(cosine_similarity(query.reshape(1, -1), candidate_vector.reshape(1, -1))[0, 0])
            if not np.isfinite(similarity):
                continue
            matches.append({
                "submission_id": submission_id,
                "score": similarity,
                "vector": candidate_vector,
                "text": item.get("text"),
                "metadata": item["metadata"],
            })
        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches[: max(1, int(top_k))]


class ContradictionDetector:
    ACTIVE_ISSUE_STATES = {
        "flooded": "flooded",
        "flooding": "flooded",
        "waterlogging": "waterlogging",
        "broken": "broken",
        "damaged": "damaged",
        "leaking": "leaking",
        "blocked": "blocked",
        "overflowing": "overflowing",
        "accumulating": "accumulating",
        "exposed": "exposed",
        "obstructed": "obstructed",
        "pothole": "pothole",
        "garbage": "garbage",
        "streetlight": "streetlight",
        "road": "road",
    }
    RESOLVED_ISSUE_STATES = {
        "repaired": "repaired",
        "fixed": "fixed",
        "restored": "restored",
        "cleared": "cleared",
        "removed": "removed",
        "resolved": "resolved",
        "no longer": "no longer",
        "no more": "no more",
        "no issue": "no issue",
        "not present": "not present",
        "has been repaired": "has been repaired",
        "has been fixed": "has been fixed",
        "already repaired": "already repaired",
        "issue resolved": "issue resolved",
    }
    HISTORICAL_MARKERS = ("used to", "previously", "in the past", "yesterday", "last week", "old")

    @staticmethod
    def normalize_issue_term(value: str | None) -> str:
        return normalize_text(value or "")

    @classmethod
    def extract_issue_state(cls, text: str | None) -> dict[str, Any]:
        normalized = normalize_text(text)
        if not normalized:
            return {"issue_terms": [], "state": "unknown", "negated": False, "resolution_indicators": []}

        issue_terms: list[str] = []
        for key in sorted(cls.ACTIVE_ISSUE_STATES, key=len, reverse=True):
            if re.search(rf"\b{re.escape(key)}\b", normalized):
                issue_terms.append(key)

        resolution_indicators: list[str] = []
        for key in sorted(cls.RESOLVED_ISSUE_STATES, key=len, reverse=True):
            pattern = key.replace(" ", r"\s+")
            if re.search(rf"\b{pattern}\b", normalized):
                resolution_indicators.append(key)

        negation_text = re.sub(
            r"\b(?:not|does not|doesn't|isn't|is not)\s+(?:working|functional|functioning)\b",
            "",
            normalized,
        )
        negated = bool(re.search(r"\b(no|not|never|nothing|without)\b", negation_text))
        negated = negated and not bool(re.search(r"\bno\s+(?:streetlight|street light)\s+working\b", normalized))
        if resolution_indicators:
            return {"issue_terms": issue_terms, "state": "resolved", "negated": False, "resolution_indicators": resolution_indicators}
        if issue_terms:
            state = "historical" if any(marker in normalized for marker in cls.HISTORICAL_MARKERS) else "active"
            return {"issue_terms": issue_terms, "state": state, "negated": negated, "resolution_indicators": []}
        return {"issue_terms": [], "state": "unknown", "negated": negated, "resolution_indicators": resolution_indicators}

    @classmethod
    def state_compatibility(cls, left_state: str | None, right_state: str | None) -> float | None:
        if left_state in {None, "unknown"} or right_state in {None, "unknown"}:
            return None
        if left_state == right_state:
            return 1.0
        return 0.0

    @classmethod
    def has_explicit_contradiction(cls, left_text: str | None, right_text: str | None, left_issue: str | None, right_issue: str | None) -> bool:
        left_state = cls.extract_issue_state(left_text)
        right_state = cls.extract_issue_state(right_text)
        if left_state["state"] == "unknown" or right_state["state"] == "unknown":
            return False
        if left_state["state"] == right_state["state"]:
            return False
        if left_issue and right_issue and normalize_text(left_issue) == normalize_text(right_issue):
            return True
        overlap = set(cls.normalize_issue_term(left_issue).split()) & set(cls.normalize_issue_term(right_issue).split())
        if not overlap and not left_state["issue_terms"] and not right_state["issue_terms"]:
            return False
        return True

    @classmethod
    def evaluate(cls, current_text: str | None, candidate_text: str | None, current_issue: str | None, candidate_issue: str | None) -> dict[str, Any]:
        left_state = cls.extract_issue_state(current_text)
        right_state = cls.extract_issue_state(candidate_text)
        state_compat = cls.state_compatibility(left_state["state"], right_state["state"])
        issue_negated = left_state["negated"] or right_state["negated"]
        explicit_state_contradiction = state_compat == 0.0
        reason_codes: list[str] = []
        negative_evidence: list[str] = []

        if explicit_state_contradiction:
            reason_codes.append("STATE_CONFLICT")
            negative_evidence.append("Explicit issue-state contradiction")
        if issue_negated:
            reason_codes.append("ISSUE_NEGATED")
            negative_evidence.append("Candidate contains a negation or resolution of the reported issue")
        if left_state["state"] == "resolved" or right_state["state"] == "resolved":
            reason_codes.append("ISSUE_RESOLVED")
            negative_evidence.append("Candidate explicitly describes the issue as resolved")

        return {
            "state_compatibility": state_compat,
            "state_conflict": explicit_state_contradiction,
            "explicit_state_contradiction": explicit_state_contradiction,
            "issue_negated": issue_negated,
            "issue_state_current": left_state,
            "issue_state_candidate": right_state,
            "negative_evidence": negative_evidence,
            "reason_codes": reason_codes,
        }


class IssueNormalizer:
    """Deterministic canonicalization for civic issue categories."""

    CANONICAL_PATTERNS: dict[str, tuple[str, ...]] = {
        "pothole": (
            "pothole",
            "potholes",
            "road pothole",
            "damaged road",
            "road damage",
            "gaddha",
            "गड्ढा",
            "सड़क में गड्ढा",
        ),
        "waterlogging": (
            "waterlogging",
            "water logged",
            "waterlogged",
            "flooded road",
            "road flooding",
            "standing water",
            "water accumulation",
            "जलभराव",
            "पानी जमा",
            "water logging",
            "road is flooded",
            "flooded",
            "flooding",
        ),
        "streetlight_failure": (
            "streetlight",
            "street light",
            "broken streetlight",
            "non-functional street light",
            "faulty street light",
            "light pole not working",
            "street light not working",
            "streetlight failure",
            "streetlight broken",
            "street light broken",
            "stree light",
        ),
        "garbage_accumulation": (
            "garbage",
            "trash",
            "litter",
            "solid waste",
            "garbage accumulation",
            "waste accumulation",
            "कूड़ा",
            "कचरा",
        ),
        "drainage_blockage": (
            "blocked drain",
            "choked drain",
            "drainage blockage",
            "blocked drainage",
            "drainage blocked",
            "clogged drain",
        ),
    }

    ISSUE_FAMILIES: dict[str, str] = {
        "pothole": "road_infrastructure",
        "waterlogging": "water_management",
        "streetlight_failure": "street_lighting",
        "garbage_accumulation": "solid_waste",
        "drainage_blockage": "water_management",
    }

    @classmethod
    def canonicalize(cls, value: str | None) -> str | None:
        text = normalize_text(value or "")
        if not text:
            return None
        for canonical, patterns in cls.CANONICAL_PATTERNS.items():
            for pattern in patterns:
                if re.search(rf"\b{re.escape(pattern.replace(' ', r'\s+'))}\b", text):
                    return canonical
                if pattern in text:
                    return canonical
        return None

    @classmethod
    def issue_family(cls, value: str | None) -> str | None:
        canonical = cls.canonicalize(value)
        return cls.ISSUE_FAMILIES.get(canonical, canonical)


class DuplicateDecisionEngine:
    def __init__(self) -> None:
        self.duplicate_threshold = float(getattr(settings, "DUPLICATE_SCORE_THRESHOLD", 0.82))
        self.possible_threshold = float(getattr(settings, "POSSIBLE_DUPLICATE_SCORE_THRESHOLD", 0.62))

    def decide(self, ranked_candidates: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        if not ranked_candidates:
            return "no_candidates", {
                "threshold": self.duplicate_threshold,
                "decision": "no_candidates",
                "reason": "No plausible candidates were retrieved from the repository.",
            }

        best = ranked_candidates[0]
        score = float(best.get("decision_score", best.get("score", 0.0)))
        contradiction = bool(
            best.get("explicit_state_contradiction")
            or best.get("issue_negated")
            or best.get("issue_mismatch")
            or best.get("geo_conflict")
        )
        if best.get("explicit_state_contradiction"):
            return "not_duplicate", {
                "threshold": self.duplicate_threshold,
                "decision": "not_duplicate",
                "reason": "The best candidate contains an explicit state contradiction that overrides text similarity.",
            }
        if bool(best.get("issue_mismatch")):
            return "not_duplicate", {
                "threshold": self.duplicate_threshold,
                "decision": "not_duplicate",
            "reason": "The candidate describes a different canonical issue, so it cannot be the same incident.",
            }
        if score >= self.duplicate_threshold and not contradiction:
            return "duplicate", {
                "threshold": self.duplicate_threshold,
                "decision": "duplicate",
                "reason": "The best candidate exceeded the duplicate threshold with strong supporting evidence and no major contradiction.",
            }
        if score >= self.possible_threshold and not contradiction:
            return "possible_duplicate", {
                "threshold": self.possible_threshold,
                "decision": "possible_duplicate",
                "reason": "The best candidate fell in the review band but not enough evidence existed for a hard duplicate decision.",
            }
        return "not_duplicate", {
            "threshold": self.duplicate_threshold,
            "decision": "not_duplicate",
            "reason": "The best candidate did not exceed the configured threshold or was offset by contradiction evidence.",
        }


class DuplicateService:
    """Deterministic duplicate detection service with an extensible retrieval/ranking pipeline."""

    def __init__(self) -> None:
        self.repository = EmbeddingRepository()
        self.decision_engine = DuplicateDecisionEngine()
        self._submission_index: dict[str, SubmissionContext] = {}

    def reset(self) -> None:
        self.repository.reset()
        self._submission_index.clear()

    @staticmethod
    def _extract_text(submission: SubmissionContext) -> str:
        analysis = getattr(submission.description, "analysis", None)
        if analysis is not None and getattr(analysis, "complaint", None):
            return str(analysis.complaint)
        return ""

    @staticmethod
    def _complaint_issue_hint(complaint: str | None) -> str | None:
        text = normalize_text(complaint or "")
        if not text:
            return None
        if any(term in text for term in ["streetlight", "street light", "light pole", "electric", "broken light"]):
            return "streetlight"
        if any(term in text for term in ["waterlogging", "water logged", "flooded", "flooding", "standing water", "waterlogged"]):
            return "waterlogging"
        if any(term in text for term in ["pothole", "damaged road", "road damage"]):
            return "pothole"
        if any(term in text for term in ["garbage", "solid waste", "trash", "litter"]):
            return "garbage"
        if any(term in text for term in ["drainage", "blocked drain", "choked drain"]):
            return "drainage"
        return None

    @staticmethod
    def _extract_issue_fields(submission: SubmissionContext) -> tuple[str | None, str | None, str | None, str | None, str | None]:
        analysis = getattr(submission.description, "analysis", None)
        category = getattr(analysis, "primary_category", None) if analysis is not None else None
        issue = getattr(analysis, "problem_type", None) if analysis is not None else None
        domain = getattr(analysis, "domain", None) if analysis is not None else None
        location = getattr(analysis, "location", None) if analysis is not None else None
        landmark = getattr(analysis, "landmark", None) if analysis is not None else None

        complaint = DuplicateService._extract_text(submission)
        complaint_issue = IssueNormalizer.canonicalize(complaint) or DuplicateService._complaint_issue_hint(complaint)
        if complaint_issue:
            issue = complaint_issue
            if complaint_issue in {"waterlogging", "drainage_blockage"}:
                category = category or "Water / Drainage"
                domain = domain or "Water / Drainage"
            elif complaint_issue == "streetlight_failure":
                category = category or "Street Lighting"
                domain = domain or "Street Lighting"
            elif complaint_issue == "pothole":
                category = category or "Road / Urban Infrastructure"
                domain = domain or "Road / Urban Infrastructure"
            elif complaint_issue == "garbage_accumulation":
                category = category or "Solid Waste Management"
                domain = domain or "Solid Waste Management"

        fusion = getattr(submission, "fusion", None)
        if fusion is not None:
            fused_category = getattr(fusion, "category", None)
            if fused_category is not None:
                category = fused_category.value or category
                if not complaint_issue:
                    issue = IssueNormalizer.canonicalize(fused_category.issue_type) or fused_category.issue_type or issue
                domain = fused_category.domain or domain
            loc_context = getattr(fusion, "location_context", None)
            if loc_context is not None:
                location = loc_context.nlp_location or location
                landmark = loc_context.nlp_landmark or landmark
        return category, issue, domain, location, landmark

    def _canonical_text(self, submission: SubmissionContext, include_summary: bool = True) -> str:
        complaint = self._extract_text(submission)
        category, issue, domain, location, landmark = self._extract_issue_fields(submission)
        severity = None
        fusion = getattr(submission, "fusion", None)
        if fusion is not None:
            severity = getattr(getattr(fusion, "severity", None), "value", None)

        fragments: list[str] = []
        if category:
            fragments.append(f"{category} civic issue")
        if issue:
            fragments.append(f"{issue} issue")
        if domain:
            fragments.append(f"{domain} domain")
        if location:
            fragments.append(f"near {location}")
        if landmark:
            fragments.append(f"at {landmark}")
        if severity:
            fragments.append(f"severity {severity.lower()}")
        if complaint:
            fragments.append(complaint)

        combined = " ".join(part for part in fragments if part).strip()
        if not include_summary:
            return normalize_text(complaint)
        return normalize_text(combined) if combined else normalize_text(complaint)

    def _issue_text(self, submission: SubmissionContext) -> str:
        category, issue, domain, location, landmark = self._extract_issue_fields(submission)
        return normalize_text(" ".join(part for part in [category, issue, domain, location, landmark] if part))

    def _embedding_payloads(self, submission: SubmissionContext) -> dict[str, str]:
        payloads = {
            "complaint_semantic": self._canonical_text(submission),
            "issue_semantic": self._issue_text(submission),
        }
        category_value = None
        fusion = getattr(submission, "fusion", None)
        if fusion is not None:
            category_value = getattr(getattr(fusion, "category", None), "value", None)
        payloads["category_semantic"] = normalize_text(category_value or self._issue_text(submission))
        return payloads

    def _build_embeddings(self, submission: SubmissionContext) -> dict[str, np.ndarray | None]:
        payloads = self._embedding_payloads(submission)
        texts = [value for value in payloads.values() if value]
        for bucket in self.repository._records.values():
            for item in bucket.values():
                text = item.get("text")
                if text:
                    texts.append(text)
        texts = list(dict.fromkeys(texts))
        if not texts:
            return {"complaint_semantic": None, "issue_semantic": None, "category_semantic": None}
        self.repository.provider.fit(texts)
        return {key: self.repository.provider.embed_text(value) for key, value in payloads.items()}

    @staticmethod
    def _token_set(text: str | None) -> set[str]:
        if not text:
            return set()
        return {token for token in normalize_text(text).split() if token}

    @staticmethod
    def _lexical_similarity(left: str | None, right: str | None) -> float:
        left_tokens = DuplicateService._token_set(left)
        right_tokens = DuplicateService._token_set(right)
        if not left_tokens or not right_tokens:
            return 0.0
        union = left_tokens | right_tokens
        if not union:
            return 0.0
        overlap = left_tokens & right_tokens
        return len(overlap) / len(union)

    @staticmethod
    def _distance_meters(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        earth_radius = 6371000.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lng2 - lng1)
        a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        return 2 * earth_radius * math.asin(math.sqrt(a))

    def _category_compatibility(self, current: SubmissionContext, candidate: SubmissionContext) -> tuple[float | None, str]:
        current_fusion = getattr(current, "fusion", None)
        candidate_fusion = getattr(candidate, "fusion", None)
        if current_fusion is None or candidate_fusion is None:
            return None, "unavailable"

        current_category = getattr(current_fusion, "category", None)
        candidate_category = getattr(candidate_fusion, "category", None)
        if current_category is None or candidate_category is None:
            return None, "unavailable"

        if normalize_text(current_category.value) and normalize_text(current_category.value) == normalize_text(candidate_category.value):
            return 1.0, "match"
        if current_category.domain and candidate_category.domain and normalize_text(current_category.domain) == normalize_text(candidate_category.domain):
            return 0.6, "partial_match"
        if current_category.issue_type and candidate_category.issue_type:
            left = normalize_text(current_category.issue_type)
            right = normalize_text(candidate_category.issue_type)
            if left == right:
                return 0.9, "match"
            if left in right or right in left:
                return 0.6, "partial_match"
        return 0.0, "mismatch"

    def _issue_compatibility_details(self, current: SubmissionContext, candidate: SubmissionContext) -> dict[str, Any]:
        _, current_issue, _, _, _ = self._extract_issue_fields(current)
        _, candidate_issue, _, _, _ = self._extract_issue_fields(candidate)
        current_id = IssueNormalizer.canonicalize(current_issue or "") or current_issue
        candidate_id = IssueNormalizer.canonicalize(candidate_issue or "") or candidate_issue
        if not current_id or not candidate_id:
            return {"score": None, "state": "unavailable", "current_issue": None, "candidate_issue": None, "reason": "Issue information is missing."}

        normalized_current = normalize_text(current_id)
        normalized_candidate = normalize_text(candidate_id)
        if normalized_current == normalized_candidate:
            return {"score": 1.0, "state": "match", "current_issue": current_id, "candidate_issue": candidate_id, "reason": f"Both complaints map to the canonical issue '{current_id}'."}

        current_family = IssueNormalizer.issue_family(current_id)
        candidate_family = IssueNormalizer.issue_family(candidate_id)
        if current_family and candidate_family and current_family == candidate_family:
            return {"score": 0.55, "state": "partial_match", "current_issue": current_id, "candidate_issue": candidate_id, "reason": f"Both complaints share the same issue family '{current_family}'."}

        return {"score": 0.0, "state": "mismatch", "current_issue": current_id, "candidate_issue": candidate_id, "reason": "The canonical issues are explicitly different."}

    def _issue_compatibility(self, current: SubmissionContext, candidate: SubmissionContext) -> tuple[float, str]:
        details = self._issue_compatibility_details(current, candidate)
        score = float(details["score"]) if details["score"] is not None else 0.0
        return score, details["state"]

    def _geo_similarity(self, current: SubmissionContext, candidate: SubmissionContext) -> tuple[float | None, bool]:
        if current.location is None or candidate.location is None:
            return None, False

        radius = float(getattr(settings, "DUPLICATE_GEO_RADIUS_METERS", 250.0))
        distance = self._distance_meters(current.location.lat, current.location.lng, candidate.location.lat, candidate.location.lng)
        if distance > radius:
            similarity = 0.0
            return float(similarity), True
        similarity = max(0.0, 1.0 - (distance / radius))
        return float(similarity), True

    @staticmethod
    def _severity_compatibility(current: SubmissionContext, candidate: SubmissionContext) -> tuple[float, str]:
        current_fusion = getattr(current, "fusion", None)
        candidate_fusion = getattr(candidate, "fusion", None)
        if current_fusion is None or candidate_fusion is None:
            return None, "unavailable"
        current_severity = getattr(getattr(current_fusion, "severity", None), "value", None)
        candidate_severity = getattr(getattr(candidate_fusion, "severity", None), "value", None)
        if not current_severity or not candidate_severity:
            return None, "unavailable"
        if normalize_text(current_severity) == normalize_text(candidate_severity):
            return 1.0, "match"
        return 0.5, "partial_match"

    @staticmethod
    def _visual_evidence_compatibility(current: SubmissionContext, candidate: SubmissionContext) -> tuple[float | None, bool]:
        current_image = getattr(current, "image", None)
        candidate_image = getattr(candidate, "image", None)
        current_analysis = getattr(current_image, "analysis", None) if current_image is not None else None
        candidate_analysis = getattr(candidate_image, "analysis", None) if candidate_image is not None else None
        current_fusion = getattr(current, "fusion", None)
        candidate_fusion = getattr(candidate, "fusion", None)
        current_states = getattr(current_fusion, "modality_states", None)
        candidate_states = getattr(candidate_fusion, "modality_states", None)
        if (
            current_analysis is None
            or candidate_analysis is None
            or not getattr(getattr(current_states, "vision", None), "usable", False)
            or not getattr(getattr(candidate_states, "vision", None), "usable", False)
        ):
            return None, False
        current_issues = getattr(current_analysis, "issues", [])
        candidate_issues = getattr(candidate_analysis, "issues", [])
        if not current_issues or not candidate_issues:
            return None, False
        current_issue = getattr(current_issues[0], "visual_issue_type", None)
        candidate_issue = getattr(candidate_issues[0], "visual_issue_type", None)
        if not current_issue or not candidate_issue:
            return None, False
        left = normalize_text(current_issue)
        right = normalize_text(candidate_issue)
        if left == right:
            return 1.0, True
        if left in right or right in left:
            return 0.7, True
        return 0.1, True

    def _rank_candidate(
        self,
        current: SubmissionContext,
        candidate: SubmissionContext,
        semantic_similarity: float | None,
        retrieved_by: list[str],
        candidate_text: str | None,
    ) -> dict[str, Any]:
        current_text = self._extract_text(current)
        lexical_similarity = self._lexical_similarity(self._canonical_text(current), candidate_text)
        category_score, category_state = self._category_compatibility(current, candidate)
        issue_details = self._issue_compatibility_details(current, candidate)
        issue_score = issue_details["score"]
        issue_state = issue_details["state"]
        geo_score, geo_available = self._geo_similarity(current, candidate)
        distance_meters = self._distance_meters(current.location.lat, current.location.lng, candidate.location.lat, candidate.location.lng) if current.location is not None and candidate.location is not None else None
        severity_score, severity_state = self._severity_compatibility(current, candidate)
        visual_score, visual_available = self._visual_evidence_compatibility(current, candidate)
        current_issue, candidate_issue = self._extract_issue_fields(current)[1], self._extract_issue_fields(candidate)[1]
        contradiction = ContradictionDetector.evaluate(current_text, candidate_text, current_issue, candidate_issue)
        explicit_state_contradiction = contradiction["explicit_state_contradiction"]
        issue_negated = contradiction["issue_negated"]
        state_compatibility = contradiction["state_compatibility"]
        same_location = bool(geo_available and geo_score is not None and geo_score >= 0.6)
        issue_mismatch = bool(issue_state == "mismatch" or (issue_score is not None and issue_score < 0.35 and issue_state != "unavailable"))
        geo_conflict = bool(geo_available and geo_score is not None and geo_score < 0.2)
        geo_state = (
            "unknown" if not geo_available
            else "nearby" if geo_score is not None and geo_score >= 0.6
            else "same_area" if geo_score is not None and geo_score >= 0.2
            else "distant"
        )

        provider_name = str(getattr(settings, "EMBEDDING_PROVIDER", "tfidf")).lower()
        tensor_similarity = semantic_similarity if provider_name != "tfidf" else None
        text_vector_similarity = semantic_similarity if provider_name == "tfidf" else semantic_similarity

        weights = {
            "text_vector_similarity": float(getattr(settings, "SEMANTIC_WEIGHT", 0.45)),
            "lexical_similarity": float(getattr(settings, "LEXICAL_WEIGHT", 0.10)),
            "category_compatibility": float(getattr(settings, "CATEGORY_WEIGHT", 0.10)),
            "issue_compatibility": float(getattr(settings, "ISSUE_WEIGHT", 0.20)),
            "geo_similarity": float(getattr(settings, "GEO_WEIGHT", 0.15)),
            "visual_evidence_compatibility": float(getattr(settings, "VISUAL_WEIGHT", 0.05)),
            "state_compatibility": 0.10,
        }

        feature_map = {
            "text_vector_similarity": {"value": text_vector_similarity, "available": text_vector_similarity is not None},
            "lexical_similarity": {"value": lexical_similarity, "available": lexical_similarity is not None},
            "category_compatibility": {"value": category_score, "available": category_score is not None},
            "issue_compatibility": {"value": issue_score, "available": issue_score is not None},
            "geo_similarity": {"value": geo_score, "available": geo_score is not None},
            "temporal_similarity": {"value": None, "available": False},
            "visual_evidence_compatibility": {"value": visual_score, "available": visual_score is not None},
            "state_compatibility": {"value": state_compatibility, "available": state_compatibility is not None},
            "severity_compatibility": {"value": severity_score, "available": severity_state != "unavailable" and severity_score is not None},
        }

        available_weights = [weights[name] for name, feature in feature_map.items() if feature["available"] and name in weights]
        if not available_weights:
            total_score = 0.0
        else:
            total_score = sum(
                feature_map[name]["value"] * weights[name]
                for name, feature in feature_map.items()
                if feature["available"] and name in weights
            ) / sum(available_weights)
        total_score = float(np.clip(total_score, 0.0, 1.0))

        contradiction_penalty = 0.0
        contradiction_breakdown = {
            "issue_mismatch": 0.0,
            "state_conflict": 0.0,
            "geo_conflict": 0.0,
            "negation": 0.0,
        }
        negative_evidence: list[str] = []
        reason_codes: list[str] = []
        if explicit_state_contradiction:
            contradiction_penalty += 0.75
            contradiction_breakdown["state_conflict"] = 0.75
            negative_evidence.append("Explicit issue-state contradiction")
            reason_codes.append("STATE_CONFLICT")
        if issue_negated:
            contradiction_penalty += 0.75
            contradiction_breakdown["negation"] = 0.75
            negative_evidence.append("The reported issue is explicitly negated.")
            reason_codes.append("NEGATED_ISSUE")
        if issue_mismatch:
            contradiction_penalty += 0.35
            contradiction_breakdown["issue_mismatch"] = 0.35
            negative_evidence.append("The canonical issues are different.")
            reason_codes.append("ISSUE_MISMATCH")
        if geo_conflict:
            contradiction_penalty += 0.20
            contradiction_breakdown["geo_conflict"] = 0.20
            negative_evidence.append("Geographic evidence strongly conflicts")
            reason_codes.append("GEO_CONFLICT")
        if category_state == "mismatch":
            negative_evidence.append("Category mismatch")
            reason_codes.append("CATEGORY_MISMATCH")
        if visual_score is None:
            reason_codes.append("MISSING_STRUCTURED_VISUAL_EVIDENCE")
        if not geo_available:
            reason_codes.append("MISSING_GEOGRAPHIC_EVIDENCE")
        decision_score = float(np.clip(total_score - contradiction_penalty, 0.0, 1.0))

        explanation: list[str] = []
        if text_vector_similarity is not None and text_vector_similarity >= 0.8:
            explanation.append(f"High text-vector similarity: {text_vector_similarity:.2f}")
            reason_codes.append("HIGH_TEXT_SIMILARITY")
        elif text_vector_similarity is not None and text_vector_similarity >= 0.5:
            explanation.append(f"Moderate text-vector similarity: {text_vector_similarity:.2f}")
        if issue_state == "match":
            explanation.append("Issue type is compatible.")
            reason_codes.append("SAME_ISSUE_TYPE")
        elif issue_state == "partial_match":
            explanation.append("Issue type is partially compatible.")
        elif issue_mismatch:
            explanation.append("The complaints describe different canonical issues.")
        if same_location:
            explanation.append("Geographic proximity supports this candidate.")
            reason_codes.append("GEO_PROXIMITY")
        elif geo_available and geo_score is not None:
            explanation.append("Geographic evidence is weak or distant.")
        else:
            explanation.append("Geographic evidence is unavailable.")
        if visual_score is None:
            explanation.append("No structured visual evidence is available.")
        if issue_negated:
            explanation.append("The complaint explicitly indicates that the issue is not present.")
        if explicit_state_contradiction:
            explanation.append(
                f"The current complaint is {contradiction['issue_state_current']['state']} while the candidate is {contradiction['issue_state_candidate']['state']}."
            )
        if category_state == "match":
            explanation.append("Category compatibility is strong.")
            reason_codes.append("SAME_CATEGORY")
        elif category_state == "mismatch":
            explanation.append("Category compatibility is weak or conflicting.")
        if visual_available and visual_score is not None and visual_score >= 0.6:
            explanation.append("Structured visual evidence is compatible.")
            reason_codes.append("VISUAL_EVIDENCE_MATCH")
        if not explanation:
            explanation.append("Insufficient evidence for a confident duplicate decision.")
            reason_codes.append("INSUFFICIENT_EVIDENCE")

        return {
            "submission_id": candidate.id,
            "score": round(total_score, 4),
            "retrieval_score": round(float(semantic_similarity), 4) if semantic_similarity is not None else None,
            "ranking_score": round(total_score, 4),
            "decision_score": round(decision_score, 4),
            "positive_score": total_score,
            "contradiction_penalty": contradiction_penalty,
            "contradiction_breakdown": {
                **contradiction_breakdown,
                "total": contradiction_penalty,
            },
            "current_state": contradiction["issue_state_current"]["state"],
            "candidate_state": contradiction["issue_state_candidate"]["state"],
            "current_negated": contradiction["issue_state_current"]["negated"],
            "candidate_negated": contradiction["issue_state_candidate"]["negated"],
            "explicit_state_contradiction": explicit_state_contradiction,
            "state_compatibility": state_compatibility,
            "issue_mismatch": issue_mismatch,
            "same_location": same_location,
            "geo_conflict": geo_conflict,
            "geo_state": geo_state,
            "missing_evidence": [
                label for label, available in (
                    ("text vector evidence", semantic_similarity is not None),
                    ("geographic evidence", geo_available),
                    ("structured visual evidence", visual_available),
                    ("issue state evidence", contradiction["issue_state_current"]["state"] != "unknown" and contradiction["issue_state_candidate"]["state"] != "unknown"),
                ) if not available
            ],
            "negative_evidence": negative_evidence,
            "reason_codes": reason_codes,
            "retrieved_by": retrieved_by,
            "signals": {
                "text_vector_similarity": text_vector_similarity,
                "semantic_similarity": tensor_similarity,
                "lexical_similarity": lexical_similarity,
                "category_compatibility": category_score,
                "issue_compatibility": issue_score,
                "distance_meters": distance_meters,
                "geo_similarity": geo_score,
                "temporal_similarity": None,
                "visual_evidence_compatibility": visual_score,
                "visual_similarity": None,
                "state_compatibility": state_compatibility,
                "geo_state": geo_state,
                "current_state": contradiction["issue_state_current"]["state"],
                "candidate_state": contradiction["issue_state_candidate"]["state"],
                "current_negated": contradiction["issue_state_current"]["negated"],
                "candidate_negated": contradiction["issue_state_candidate"]["negated"],
                "canonical_issue_current": IssueNormalizer.canonicalize(current_issue or "") or current_issue,
                "canonical_issue_candidate": IssueNormalizer.canonicalize(candidate_issue or "") or candidate_issue,
            },
            "explanation": explanation,
            "candidate_text": candidate_text,
        }

    def _store_submission(self, submission: SubmissionContext) -> None:
        self._submission_index[str(submission.id)] = submission
        payloads = self._embedding_payloads(submission)
        embeddings = self._build_embeddings(submission)
        for embedding_type, vector in embeddings.items():
            if vector is None:
                continue
            self.repository.save(
                submission.id,
                embedding_type,
                vector,
                {
                    "provider": settings.EMBEDDING_PROVIDER,
                    "model": settings.EMBEDDING_MODEL,
                    "dimension": int(np.asarray(vector).size),
                    "version": settings.EMBEDDING_VERSION,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                payloads.get(embedding_type),
            )
        self.repository.refresh_vectors()

    def _candidate_ids_from_search(self, current: SubmissionContext, top_k: int) -> dict[str, list[str]]:
        query_embeddings = self._build_embeddings(current)
        complaint_vector = query_embeddings.get("complaint_semantic")
        if complaint_vector is None:
            return {"semantic": [], "issue": [], "category": [], "geo": [], "metadata": []}

        semantic = self.repository.search(complaint_vector, top_k=top_k, embedding_type="complaint_semantic")
        issue_vector = query_embeddings.get("issue_semantic")
        issue = self.repository.search(issue_vector, top_k=top_k, embedding_type="issue_semantic") if issue_vector is not None else []
        category_vector = query_embeddings.get("category_semantic")
        category = self.repository.search(category_vector, top_k=top_k, embedding_type="category_semantic") if category_vector is not None else []

        semantic_ids = [match["submission_id"] for match in semantic if match["submission_id"] != current.id]
        issue_ids = [match["submission_id"] for match in issue if match["submission_id"] != current.id]
        category_ids = [match["submission_id"] for match in category if match["submission_id"] != current.id]

        geo_ids: list[str] = []
        if current.location is not None:
            for submission_id, submission in self._submission_index.items():
                if submission_id == current.id or submission.location is None:
                    continue
                distance = self._distance_meters(current.location.lat, current.location.lng, submission.location.lat, submission.location.lng)
                if distance <= float(getattr(settings, "DUPLICATE_GEO_RADIUS_METERS", 250.0)):
                    geo_ids.append(submission_id)

        metadata_ids: list[str] = []
        if current.description and getattr(current.description, "analysis", None) is not None:
            current_category, current_issue, current_domain, _, _ = self._extract_issue_fields(current)
            for submission_id, submission in self._submission_index.items():
                if submission_id == current.id:
                    continue
                candidate_category, candidate_issue, candidate_domain, _, _ = self._extract_issue_fields(submission)
                compatibility = False
                if current_category and candidate_category and normalize_text(current_category) == normalize_text(candidate_category):
                    compatibility = True
                if current_issue and candidate_issue and normalize_text(current_issue) == normalize_text(candidate_issue):
                    compatibility = True
                if current_domain and candidate_domain and normalize_text(current_domain) == normalize_text(candidate_domain):
                    compatibility = True
                if compatibility:
                    metadata_ids.append(submission_id)

        return {
            "semantic": semantic_ids,
            "issue": issue_ids,
            "category": category_ids,
            "geo": geo_ids,
            "metadata": metadata_ids,
        }

    def process(self, submission: SubmissionContext) -> SubmissionContext:
        if getattr(submission, "description", None) is None:
            return submission

        if not self._extract_text(submission).strip():
            submission.duplicate = {
                "status": "insufficient_data",
                "is_duplicate": False,
                "best_match": None,
                "candidates": [],
                "signals": {
                    "semantic_similarity": None,
                    "lexical_similarity": None,
                    "category_compatibility": None,
                    "issue_compatibility": None,
                    "geo_similarity": None,
                    "temporal_similarity": None,
                    "visual_similarity": None,
                },
                "decision": {
                    "threshold": self.decision_engine.duplicate_threshold,
                    "decision": "insufficient_data",
                    "reason": "Submission description is empty or whitespace only.",
                },
                "explanation": ["No complaint text was available for duplicate analysis."],
                "provenance": {"engine": "deterministic_duplicate_engine", "version": "1.2"},
            }
            return submission

        self._store_submission(submission)

        query_embeddings = self._build_embeddings(submission)
        complaint_vector = query_embeddings.get("complaint_semantic")
        if complaint_vector is None:
            submission.duplicate = {
                "status": "unavailable",
                "is_duplicate": False,
                "best_match": None,
                "candidates": [],
                "signals": {
                    "semantic_similarity": None,
                    "lexical_similarity": None,
                    "category_compatibility": None,
                    "issue_compatibility": None,
                    "geo_similarity": None,
                    "temporal_similarity": None,
                    "visual_similarity": None,
                },
                "decision": {
                    "threshold": self.decision_engine.duplicate_threshold,
                    "decision": "unavailable",
                    "reason": "Embedding generation failed or produced no valid vector.",
                },
                "explanation": ["The embedding provider returned no usable vectors."],
                "provenance": {"engine": "deterministic_duplicate_engine", "version": "1.2", "text_provider": settings.EMBEDDING_PROVIDER, "text_model": settings.EMBEDDING_MODEL, "visual_mode": "structured_cv_evidence", "decision_mode": "rule_based"},
            }
            return submission

        candidate_ids_by_source = self._candidate_ids_from_search(submission, top_k=int(getattr(settings, "DUPLICATE_TOP_K", 10)))
        candidate_ids: dict[str, list[str]] = defaultdict(list)
        for source, ids in candidate_ids_by_source.items():
            for candidate_id in ids:
                candidate_ids[candidate_id].append(source)

        filtered_candidates: list[SubmissionContext] = []
        for candidate_id in sorted(candidate_ids):
            if candidate_id == submission.id:
                continue
            candidate_submission = self._submission_index.get(str(candidate_id))
            if candidate_submission is None:
                continue
            filtered_candidates.append(candidate_submission)

        if not filtered_candidates:
            submission.duplicate = {
                "status": "no_candidates",
                "is_duplicate": False,
                "best_match": None,
                "candidates": [],
                "signals": {
                    "semantic_similarity": None,
                    "lexical_similarity": None,
                    "category_compatibility": None,
                    "issue_compatibility": None,
                    "geo_similarity": None,
                    "temporal_similarity": None,
                    "visual_similarity": None,
                },
                "decision": {
                    "threshold": self.decision_engine.duplicate_threshold,
                    "decision": "no_candidates",
                    "reason": "No existing complaints were retrieved as likely duplicates.",
                },
                "explanation": ["The repository was empty or no candidate passed the retrieval filters."],
                "provenance": {"engine": "deterministic_duplicate_engine", "version": "1.2", "text_provider": settings.EMBEDDING_PROVIDER, "text_model": settings.EMBEDDING_MODEL, "visual_mode": "structured_cv_evidence", "decision_mode": "rule_based"},
            }
            return submission

        ranked: list[dict[str, Any]] = []
        semantic_matches = self.repository.search(complaint_vector, top_k=10, embedding_type="complaint_semantic")
        for candidate_submission in filtered_candidates:
            match = next((item for item in semantic_matches if item["submission_id"] == candidate_submission.id), None)
            semantic_similarity = float(match["score"]) if match else None
            candidate_rank = self._rank_candidate(
                current=submission,
                candidate=candidate_submission,
                semantic_similarity=semantic_similarity,
                retrieved_by=list(dict.fromkeys(candidate_ids.get(candidate_submission.id, []))),
                candidate_text=self._canonical_text(candidate_submission),
            )
            ranked.append(candidate_rank)

        ranked.sort(key=lambda item: item["score"], reverse=True)
        for index, item in enumerate(ranked, start=1):
            item["rank"] = index

        final_status, decision = self.decision_engine.decide(ranked)
        best = ranked[0]
        signals = {
            "semantic_similarity": best["signals"].get("semantic_similarity") if str(getattr(settings, "EMBEDDING_PROVIDER", "tfidf")).lower() != "tfidf" else None,
            "text_vector_similarity": best["signals"].get("text_vector_similarity"),
            "lexical_similarity": best["signals"].get("lexical_similarity"),
            "category_compatibility": best["signals"].get("category_compatibility"),
            "issue_compatibility": best["signals"].get("issue_compatibility"),
            "distance_meters": best["signals"].get("distance_meters"),
            "geo_similarity": best["signals"].get("geo_similarity"),
            "temporal_similarity": None,
            "visual_similarity": None,
            "visual_evidence_compatibility": best["signals"].get("visual_evidence_compatibility"),
            "state_compatibility": best["signals"].get("state_compatibility"),
            "current_state": best.get("current_state"),
            "candidate_state": best.get("candidate_state"),
            "geo_state": best.get("geo_state"),
            "canonical_issue_current": best["signals"].get("canonical_issue_current"),
            "canonical_issue_candidate": best["signals"].get("canonical_issue_candidate"),
        }
        positive_evidence: list[str] = []
        negative_evidence: list[str] = []
        reason_codes: list[str] = []
        text_similarity = signals.get("text_vector_similarity") or signals.get("semantic_similarity")
        if text_similarity is not None and text_similarity >= 0.8:
            positive_evidence.append("High text-vector similarity")
            reason_codes.append("HIGH_TEXT_SIMILARITY")
        elif text_similarity is not None and text_similarity >= 0.5:
            positive_evidence.append("Moderate text-vector similarity")
        if signals.get("issue_compatibility") == 1.0:
            positive_evidence.append("Same issue type")
            reason_codes.append("SAME_ISSUE_TYPE")
        elif signals.get("issue_compatibility") is not None and signals["issue_compatibility"] < 0.5:
            negative_evidence.append("Issue mismatch reduces duplicate confidence")
            reason_codes.append("ISSUE_MISMATCH")
        if signals.get("category_compatibility") == 1.0:
            positive_evidence.append("Same civic category")
            reason_codes.append("SAME_CATEGORY")
        if signals.get("geo_similarity") is not None and signals["geo_similarity"] >= 0.6:
            positive_evidence.append("Nearby geographic coordinates")
            reason_codes.append("GEO_PROXIMITY")
        elif signals.get("geo_similarity") is not None and signals["geo_similarity"] < 0.2:
            negative_evidence.append("Geographic evidence conflicts with an automatic duplicate decision")
            reason_codes.append("GEO_CONFLICT")
        if best.get("explicit_state_contradiction"):
            negative_evidence.append("Explicit issue-state contradiction")
            reason_codes.append("STATE_CONFLICT")
        if "NEGATED_ISSUE" in best.get("reason_codes", []):
            negative_evidence.append("The reported issue is explicitly negated")
            reason_codes.append("NEGATED_ISSUE")
        missing_evidence = best.get("missing_evidence", [])
        for evidence_name in missing_evidence:
            reason_codes.append(f"MISSING_{evidence_name.upper().replace(' ', '_')}")
        if best.get("state_compatibility") == 0.0:
            negative_evidence.append("Issue states conflict")
            reason_codes.append("STATE_CONFLICT")
        if final_status == "duplicate":
            reason_codes.append("ABOVE_DUPLICATE_THRESHOLD")
        elif final_status == "possible_duplicate":
            reason_codes.append("ABOVE_POSSIBLE_THRESHOLD")
        if not positive_evidence:
            positive_evidence.append("Comparable issue context was retrieved")
        if not negative_evidence and final_status == "not_duplicate":
            negative_evidence.append("Evidence did not cross the configured duplicate threshold")
        reason_codes = list(dict.fromkeys(reason_codes))

        explanation_text = list(best["explanation"])
        if best.get("explicit_state_contradiction") and best.get("current_state") != best.get("candidate_state"):
            current_state = best.get("current_state")
            candidate_state = best.get("candidate_state")
            explanation_text.append(
                f"The issue states conflict: the current complaint is {current_state} while the candidate is {candidate_state}. This contradiction overrides the otherwise strong similarity."
            )

        submission.duplicate = {
            "status": final_status,
            "is_duplicate": final_status == "duplicate",
            "best_match": {
                "submission_id": best["submission_id"],
                "retrieval_score": best.get("retrieval_score", best.get("score")),
                "ranking_score": best.get("ranking_score", best.get("score")),
                "contradiction_penalty": best.get("contradiction_penalty", 0.0),
                "decision_score": best.get("decision_score", best.get("score")),
                "score": best["decision_score"],
                "rank": best.get("rank", 1),
            },
            "candidates": [
                {
                    "submission_id": item["submission_id"],
                    "retrieval_score": item.get("retrieval_score", item.get("score")),
                    "ranking_score": item.get("ranking_score", item.get("score")),
                    "contradiction_penalty": item.get("contradiction_penalty", 0.0),
                    "decision_score": item.get("decision_score", item.get("score")),
                    "score": item.get("decision_score", item.get("score")),
                    "rank": item.get("rank", 1),
                    "retrieved_by": item["retrieved_by"],
                    "signals": item["signals"],
                    "explanation": item["explanation"],
                }
                for item in ranked
            ],
            "signals": signals,
            "signal_breakdown": signals,
            "positive_evidence": positive_evidence,
            "negative_evidence": negative_evidence,
            "missing_evidence": missing_evidence,
            "reason_codes": reason_codes,
            "decision": {
                **decision,
                "score": float(best.get("decision_score", best.get("score", 0.0))),
            },
            "explanation": explanation_text,
            "provenance": {"engine": "deterministic_duplicate_engine", "version": "1.2", "text_provider": settings.EMBEDDING_PROVIDER, "text_model": settings.EMBEDDING_MODEL, "visual_mode": "structured_cv_evidence", "decision_mode": "rule_based"},
        }
        return submission


duplicate_service = DuplicateService()