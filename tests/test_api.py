from httpx import Response
import unittest

from uuid import uuid4
from app.main import app
from app.core.logs import logger
from fastapi.testclient import TestClient


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
LOCALHOST = "http://localhost:5050"


class TestAPI(unittest.TestCase):
    client = TestClient(app)

    def test_get_recommendation_success(self):
        data = self.client.post("/api/v1/talents", json=user).json()

        self.assertEqual(data["message"], "Successfully getting recommendation")
        self.assertEqual(len(data["data"]), 200)
