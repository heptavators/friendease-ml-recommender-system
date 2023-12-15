import joblib
import tensorflow as tf


def build_model():
    """
    Returns:
            built TFRS model
    """

    # singleton design pattern
    global model_obj

    if not "model_obj" in globals():
        model_obj = tf.saved_model.load("recommender_model")

    return model_obj


def build_vectorizer():
    """
    Returns:
            built vectorizers model
    """

    # singleton design pattern
    global vectorizer_tags, vectorizer_text

    if not "vectorizer_tags" in globals():
        vectorizer_tags = joblib.load("vectorizer_tags.joblib")

    if not "vectorizer_text" in globals():
        vectorizer_text = joblib.load("vectorizer_text.joblib")

    return vectorizer_tags, vectorizer_text
