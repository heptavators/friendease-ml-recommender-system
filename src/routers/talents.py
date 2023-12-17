import time
import asyncio

from logs.logger import logger
from models import recommender, schemas
from fastapi import APIRouter, Request, status

router = APIRouter(
    prefix="/api/recommendation",
    tags=["Face Verification"],
)


@router.get("/")
async def root():
    return {"message": "Endpoint for face verification"}


@router.post(
    "/talents",
    responses={
        str(status.HTTP_500_INTERNAL_SERVER_ERROR): {
            "description": "Internal server error"
        }
    },
)
async def get_talents(payload: schemas.User) -> schemas.ListTalent:
    start = time.perf_counter()

    response = recommender.get_list_talents(payload)

    end = time.perf_counter()
    logger.debug(f"Recommendation needs {end-start} seconds")

    return response
