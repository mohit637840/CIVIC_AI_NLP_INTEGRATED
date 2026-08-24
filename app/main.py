from fastapi import FastAPI

from app.api.cv import router as cv_router
from app.api.geo import router as geo_router
from app.api.nlp import router as nlp_router
from app.api.submissions import router as submissions_router

app = FastAPI(
    title="Civic AI Backend",
    version="0.1.0",
    description="Computer vision and geospatial services for civic complaints.",
)

app.include_router(cv_router)
app.include_router(geo_router)
app.include_router(nlp_router)
app.include_router(submissions_router)


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
