import unittest

from uuid import uuid4
from common import functions
from models import schemas

user = schemas.User(
    id=str(uuid4()),
    name="Aoi Todo",
    gender="L",
    birth_date="12/06/2002",
    age=21,
    location="Kalimantan Timur, Balikpapan",
    tags=[
        "Wibu",
        "Ngopi",
        "Prank",
        "Gamers",
        "Pengusaha",
    ],
    preferences="Saya ingin memiliki teman wibu untuk datang ke event cosplay bareng",
)


class TestRecommender(unittest.TestCase):
    _talents = functions.get_talents()

    def test_recommend_with_tfidf(self):
        recommended_talents = functions.get_recommendation_with_tfidf(user)
        self.assertEqual(len(recommended_talents), 100)
        self.assertTrue(
            self._talents["talent_id"].isin(recommended_talents).sum() == 100
        )

    def test_recommend_with_tfrs(self):
        recommended_talents = functions.get_recommendation_with_tfrs(user)
        self.assertEqual(len(recommended_talents), 100)
        self.assertTrue(
            self._talents["talent_id"].isin(recommended_talents).sum() == 100
        )
