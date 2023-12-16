import unittest

from uuid import uuid4
from common import functions
from models import schemas
from logs.logger import logger

user = schemas.User(
    id=str(uuid4()),
    name="Aoi Todo",
    gender="L",
    birth_date="12/06/2002",
    age=21,
    location="Jawa Timur, Surabaya",
    tags=[
        "Desain",
        "Fashion",
        "Kreatif",
        "Perfeksionis",
        "Ramah",
    ],
    preferences="Ingin memiliki teman yang bisa diajak kulineran dan staycation untuk mengelilingi Indonesia",
)


class TestRecommender(unittest.TestCase):
    _talents = functions.get_talents()

    def test_recommend_with_tfidf(self):
        recommended_talents = functions.get_recommendation_with_tfidf(user)

        # logger.debug(
        #     self._talents.loc[
        #         self._talents["talent_id"].isin(recommended_talents), ["talent_tags"]
        #     ]
        # )

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

    def test_recommend_talents(self):
        recommended_talents = functions.get_recommended_talents(user)
        talents = self._talents.set_index("talent_id")

        # logger.debug(
        #     talents.loc[
        #         recommended_talents,
        #         ["talent_location", "talent_tags"],
        #     ]
        # )
        self.assertEqual(len(recommended_talents), 200)
