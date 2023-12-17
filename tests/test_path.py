import joblib
import unittest
import pandas as pd
import tensorflow as tf

from logs.logger import logger


class TestPath(unittest.TestCase):
    def test_load_csv(self):
        df = pd.read_csv("data/talents.csv")
        self.assertIsNotNone(df)

    def test_load_vectorizer(self):
        vectorizer_tags = joblib.load("models/vectorizer_tags.joblib")
        vectorizer_text = joblib.load("models/vectorizer_text.joblib")
        self.assertIsNotNone(vectorizer_tags)
        self.assertIsNotNone(vectorizer_text)

    def test_load_tfrs(self):
        tfrs = tf.saved_model.load("models/recommender_model")
        self.assertIsNotNone(tfrs)

    def test_load_path(self):
        from common.config import config

        logger.debug(config)
        self.assertIsNotNone(config)
