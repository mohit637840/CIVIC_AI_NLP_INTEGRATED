# NLP Integration

The teammate's existing Jharkhand NLP pipeline is integrated into this FastAPI project without rewriting the trained model.

## What was added

- `app/integrations/nlp/teammate_nlp.py` — adapted copy of the teammate's existing `nlp_model.py`.
- `app/integrations/nlp/jharkhand_multilingual_model/` — trained TF-IDF vectorizer, classifier, and label encoder artifacts.
- `app/services/nlp_service.py` — thin adapter exposing `nlp_service.analyze(text)`.
- `app/api/nlp.py` — `POST /api/v1/nlp/analyze` for direct NLP testing.
- `app/schemas/nlp.py` — schema matching the current NLP output with extra fields allowed.

## Submission integration

`POST /api/v1/submissions` now runs whichever modalities are supplied:

- description -> teammate NLP
- image -> Gemini vision
- location -> existing Geo service

The response keeps the raw module outputs together and adds a typed, deterministic rule-based `fusion` result. Fusion selects vision classification above the configured confidence threshold, falls back to NLP, treats resolved Geo as authoritative for location, normalizes severity, derives categorical priority and routing from the final result, and records conflicts and decision traces. The canonical internal `SubmissionContext` then passes through duplicate, priority, and routing processors; duplicate detection remains unavailable until its dedicated engine is implemented.

Missing modalities are supported. At least one of description, image, or location must be supplied.

Fusion exposes explicit modality states (`not_provided`, `valid`, `low_confidence`, `invalid`, or `failed`), final-category priority/routing, conflict records, explainability rules, and structured `duplicate_features`. GPS fallback responses retain coordinates with `resolution_status=coordinates_only` when reverse geocoding cannot enrich administrative fields.

## Dependency note

The serialized NLP artifacts were produced with scikit-learn 1.9.0. The project therefore pins `scikit-learn==1.9.0` and adds `joblib>=1.5.3`.

After cloning, run `uv sync` and then:

```powershell
uv run uvicorn app.main:app --reload
```

If the lock file needs refreshing after the dependency change, run `uv lock` once with network access and then commit the refreshed `uv.lock`.
