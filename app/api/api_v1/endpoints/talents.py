import time

from app.core.logs import logger
from app.api import recommender
from fastapi import APIRouter
from app.schemas import User, ListTalent

router = APIRouter()


@router.post(
    "/",
)
async def get_talents(payload: User) -> ListTalent:
    start = time.perf_counter()

    response = recommender.get_list_talents(payload)

    end = time.perf_counter()
    logger.debug(f"Recommendation needs {end-start} seconds")

    return response
