import httpx
import unittest

from uuid import uuid4
from common import functions
from models import schemas
from logs.logger import logger

user = {
    "id": str(uuid4()),
    "name": "Aoi Todo",
    "gender": "L",
    "birth_date": "12/06/2002",
    "age": 21,
    "location": "Jawa Timur, Surabaya",
    "tags": [
        "Desain",
        "Fashion",
        "Kreatif",
        "Perfeksionis",
        "Ramah",
    ],
    "preferences": "Ingin memiliki teman yang bisa diajak kulineran dan staycation untuk mengelilingi Indonesia",
}
LOCALHOST = "http://127.0.0.1:8000"


class TestAPI(unittest.TestCase):
    def __fetch_json__(self, url: str) -> dict:
        with httpx.Client() as client:
            return client.get(url).json()

    def __post__(self, url: str, payload: dict) -> dict:
        with httpx.Client() as client:
            return client.post(url, json=payload).json()

    def test_app_root(self):
        data = self.__fetch_json__(f"{LOCALHOST}")
        self.assertEqual(data, {"message": "Recommendation System API"})

    def test_get_recommendation_success(self):
        data = self.__post__(f"{LOCALHOST}/api/recommendation/talents", user)
        # logger.debug(data)

        self.assertEqual(data["message"], "Successfully getting recommendation")
        self.assertEqual(len(data["data"]), 200)
