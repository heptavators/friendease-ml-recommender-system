import joblib
import tensorflow as tf

from common.config import config


def build_model():
    if "model_obj" not in build_model.__dict__:
        build_model.model_obj = tf.saved_model.load(config["model"]["tfrs"])
    return build_model.model_obj


def build_vectorizer():
    if "vectorizer_tags" not in build_vectorizer.__dict__:
        build_vectorizer.vectorizer_tags = joblib.load(
            config["model"]["vectorizer_tags"]
        )

    if "vectorizer_text" not in build_vectorizer.__dict__:
        build_vectorizer.vectorizer_text = joblib.load(
            config["model"]["vectorizer_text"]
        )

    return build_vectorizer.vectorizer_tags, build_vectorizer.vectorizer_text


index = build_model()
vectorizer_tags, vectorizer_text = build_vectorizer()
