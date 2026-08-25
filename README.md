# SIH Civic AI Backend

FastAPI backend for the civic computer-vision and geospatial components.

## Current capabilities

- `POST /api/v1/cv/analyze` — Gemini vision analysis for citizen images.
- `POST /api/v1/geo/resolve` — GPS or manual-address resolution.
- `POST /api/v1/geo/cluster` — geospatial clustering for map/hotspot support.
- `POST /api/v1/nlp/analyze` — teammate NLP model adapter.
- `POST /api/v1/submissions` — orchestration entry point for optional description + image + location.
- `GET /health` — health check.

The teammate's existing TF-IDF + LinearSVC NLP pipeline is integrated through `app/services/nlp_service.py`. The original trained model artifacts are packaged under `app/integrations/nlp/jharkhand_multilingual_model/`.

The submission endpoint runs only the modalities supplied by the citizen, preserves their raw structured outputs, and returns a deterministic rule-based fusion result. Fused priority and routing are now exposed both inside `fusion` and through the existing top-level response fields. Duplicate detection is now implemented as a deterministic retrieval → ranking → decision pipeline attached to the canonical submission context.

## Duplicate detection engine

The duplicate engine is intentionally conservative and extensible. It preserves the existing fusion flow and attaches the final duplicate result to each canonical submission without replacing NLP, CV, Geo, or rule-based fusion logic.

### Current baseline

- Representation type: `tfidf`
- Embedding provider: `TfidfVectorizer(char_wb)`
- Model family: TF-IDF text vector baseline, not a transformer semantic embedding
- Repository: in-memory repository suitable for local development and testing, not a production vector database

This is a baseline for retrieval and ranking only. It is useful for local duplicate detection and for rapid iteration while the project keeps the architecture ready for sentence-transformer or multilingual embedding providers later.

### Pipeline stages

1. Evidence extraction from the canonical submission and Fusion output.
2. Representation building from complaint text, issue text, category, domain, location, severity, and issue metadata.
3. Retrieval across complaint, issue, category, geo, and metadata signals.
4. Candidate filtering to remove self-match, malformed entries, incompatible dimensions, and empty repositories.
5. Ranking using semantic similarity, lexical overlap, issue compatibility, category compatibility, geo proximity, and structured visual evidence compatibility.
6. Deterministic decision with `duplicate`, `possible_duplicate`, `not_duplicate`, `no_candidates`, `insufficient_data`, `unavailable` states.
7. Explainable output with positive and negative evidence, deterministic reason codes, and provenance metadata.

### Important terminology and limitations

- TF-IDF is treated as a text-vector baseline. It is not claimed to be true semantic understanding.
- The system supports multilingual evaluation but does not assume word overlap is sufficient across English, Hindi, and Hinglish.
- Geo and visual evidence are supporting signals, not automatic duplicate rules.
- Missing modalities do not count as negative evidence. They are simply unavailable and the active signal set is normalized over what is available.
- Thresholds are engineering defaults and are not a calibrated probability estimate.

### Retrieval vs decision

Retrieval asks: "Could this complaint be related?"
Decision asks: "Is this complaint actually a duplicate?"

The engine keeps this distinction explicit: scores are tracked separately for retrieval, ranking, and decision output.

### Explainability

Final duplicate results include:

- `status`
- `best_match`
- `candidates`
- `signal_breakdown`
- `positive_evidence`
- `negative_evidence`
- `reason_codes`
- `decision`
- `provenance`

This makes duplicate decisions inspectable without exposing raw vectors in normal API responses.

## Fusion engine

`FusionService` receives typed NLP, vision, and resolved Geo results without depending on FastAPI. It returns a rich `FusionResult` with selected values, evidence, source attribution, modality availability, confidence metadata, conflicts, and a decision trace.

Rules are explicit and deterministic:

- Category and issue type use vision when a detected issue's confidence is at least `VISION_CATEGORY_CONFIDENCE_THRESHOLD` (default `0.75`); otherwise NLP is the fallback.
- Geo always wins location selection when successfully resolved. NLP location and landmark remain in `location_context`; disagreements are recorded as conflicts.
- Severity is normalized to `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL`. High-confidence disagreement selects the higher reliable signal; agreement is attributed to `fused`.
- Priority is categorical and derived from final severity plus available hazards, obstruction, urgency, affected people, visible scale, and public-impact evidence. No arbitrary numerical score is invented.
- Routing derives a department from the final fused category using the existing NLP taxonomy mappings. Unmapped visual categories remain `null` with an explanation.
- Model confidence is copied only from existing NLP/CV outputs. Missing confidence remains `null`; Geo's `1.0` represents authoritative validated source selection, not an ML probability.
- Missing or failed modalities produce empty/null fusion fields and do not prevent available modalities from being fused.
- `modality_states` distinguishes `not_provided`, `valid`, `low_confidence`, `invalid`, and `failed`; `modality_availability` indicates usable evidence only.
- `duplicate_features` keeps final category, issue type, domain, normalized complaint, visual description, evidence, location fields, severity, landmark, and affected objects ready for a future detector without creating embeddings.

Fusion is isolated behind `FusionService`, so a future learned model can implement the same typed input/output boundary without changing the submission API.

## Submission contract

`POST /api/v1/submissions` accepts JSON. At least one of `description`, `image_base64`/`image_url`, or `location` is required. `location` contains exactly one of `gps_coordinates` or `manual_address`.

Request example:

```json
{
	"description": "There is severe waterlogging on the road near Main Market in Ranchi.",
	"image_base64": null,
	"image_url": null,
	"location": {
		"gps_coordinates": {"lat": 23.3441, "lng": 85.3096}
	}
}
```

Text-only submission:

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/submissions -H "Content-Type: application/json" -d '{"description":"There is a pothole near the school."}'
```

Text plus GPS:

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/submissions -H "Content-Type: application/json" -d '{"description":"There is severe waterlogging near Main Market in Ranchi.","location":{"gps_coordinates":{"lat":23.3441,"lng":85.3096}}}'
```

Text plus image plus GPS (PowerShell base64 example):

```powershell
$image = [Convert]::ToBase64String([IO.File]::ReadAllBytes("issue.jpg")); $body = @{description="There is waterlogging near Main Market in Ranchi."; image_base64=$image; location=@{gps_coordinates=@{lat=23.3441; lng=85.3096}}} | ConvertTo-Json -Depth 5; curl -X POST http://127.0.0.1:8000/api/v1/submissions -H "Content-Type: application/json" -d $body
```

## Run with uv

```powershell
uv venv
uv sync
$env:GEMINI_API_KEY="YOUR_KEY"
uv run uvicorn app.main:app --reload
```

Copy `.env.example` to `.env` if preferred.
