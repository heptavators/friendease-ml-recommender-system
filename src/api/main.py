import numpy as np
import os

from fastapi import FastAPI
from src.routers import talents


os.environ["$WEB_CONCURRENCY"] = str(os.cpu_count() + 4)

app = FastAPI(
    title="Recommender System",
    description="API Recommendation System for FriendEase Application",
    version="1.0.0",
)


@app.get("/")
async def root():
    return {"message": "Recommendation System API"}


app.include_router(talents.router)
