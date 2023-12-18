import joblib
import unittest
import pandas as pd
import tensorflow as tf

from app.core.logs import logger
from app.core.config import settings


class TestPath(unittest.TestCase):
    def test_load_csv(self):
        df = pd.read_csv(settings.CONFIG["data"]["talents"])
        self.assertIsNotNone(df)

    def test_load_vectorizer(self):
        vectorizer_tags = joblib.load(settings.CONFIG["model"]["vectorizer_tags"])
        vectorizer_text = joblib.load(settings.CONFIG["model"]["vectorizer_text"])
        self.assertIsNotNone(vectorizer_tags)
        self.assertIsNotNone(vectorizer_text)

    def test_load_tfrs(self):
        tfrs = tf.saved_model.load(settings.CONFIG["model"]["tfrs"])
        self.assertIsNotNone(tfrs)

    def test_load_path(self):
        config = settings.CONFIG

        self.assertIsNotNone(config)
