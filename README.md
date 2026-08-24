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

The submission endpoint runs only the modalities supplied by the citizen, preserves their raw structured outputs, and returns a deterministic rule-based fusion result. Fused priority and routing are now exposed both inside `fusion` and through the existing top-level response fields; duplicate detection remains reserved for the next stage.

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
