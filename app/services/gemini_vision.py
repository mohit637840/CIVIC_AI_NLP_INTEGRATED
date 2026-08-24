from __future__ import annotations

from google import genai
from google.genai import types

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.schemas.vision import VisionAnalysisResult


CIVIC_VISION_PROMPT = """
You are the Computer Vision component of an AI-powered Civic Complaint Management System.
Analyze the submitted image and identify only civic issues supported by visible evidence.

Possible domains include Waste Management, Water / Drainage, Road Infrastructure,
Electricity, Public Infrastructure, Environment, Agriculture, Healthcare, Education,
Accessibility, Rural Livelihoods, Urban Development, Public Administration, and Other.

Rules:
1. Do not invent facts that cannot be seen.
2. Use the citizen description only as supporting context; do not treat it as visual proof.
3. If no civic issue is visibly present, issue_detected must be false and issues must be empty.
4. Multiple independent visible issues may be returned.
5. Severity must be based only on visible evidence.
6. Do not infer exact geographical location, duration, responsible department, or exact
   affected population from the image.
7. Provide concrete visual evidence and practical visible-condition-based actions.
8. If image quality is poor, describe that in image_quality_notes and lower confidence.
9. Confidence is model confidence, not a calibrated probability.
"""


class GeminiVisionService:
    def __init__(self) -> None:
        self.model = GEMINI_MODEL
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        if self._client is None:
            self._client = genai.Client(api_key=GEMINI_API_KEY)
        return self._client

    async def analyze(
        self,
        image_bytes: bytes,
        mime_type: str,
        user_description: str | None = None,
    ) -> VisionAnalysisResult:
        if not image_bytes:
            raise ValueError("Image is empty")

        context = ""
        if user_description:
            context = (
                "\nCitizen-provided description (supporting context only):\n"
                f"{user_description}\n"
            )

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                CIVIC_VISION_PROMPT + context,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VisionAnalysisResult,
                temperature=0.1,
            ),
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response")

        try:
            return VisionAnalysisResult.model_validate_json(response.text)
        except Exception as exc:
            raise RuntimeError("Gemini returned invalid structured vision output") from exc


vision_service = GeminiVisionService()
