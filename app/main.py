from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.api_v1.api import api_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)


@app.get("/")
def root():
    image_link = "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/ac791267-5c99-4107-b027-be55ee6efbc5/width=450/01.jpeg"

    return HTMLResponse(
        content=f'<div style="text-align:center"><img src="{image_link}" alt="Image"></div>',
        status_code=200,
    )


# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)
