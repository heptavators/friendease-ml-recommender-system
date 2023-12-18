import joblib
import tensorflow as tf

from app.core.config import settings


class ModelBuilder:
    _tfrs_obj = None
    _vectorizer_tags = None
    _vectorizer_text = None

    @staticmethod
    def get_tfrs_instance():
        if ModelBuilder._tfrs_obj is None:
            ModelBuilder._tfrs_obj = tf.saved_model.load(
                settings.CONFIG["model"]["tfrs"]
            )
        return ModelBuilder._tfrs_obj

    @staticmethod
    def get_vectorizer_instance():
        if ModelBuilder._vectorizer_tags is None:
            ModelBuilder._vectorizer_tags = joblib.load(
                settings.CONFIG["model"]["vectorizer_tags"]
            )

        if ModelBuilder._vectorizer_text is None:
            ModelBuilder._vectorizer_text = joblib.load(
                settings.CONFIG["model"]["vectorizer_text"]
            )

        return (
            ModelBuilder._vectorizer_tags,
            ModelBuilder._vectorizer_text,
        )


tfrs = ModelBuilder.get_tfrs_instance()
vectorizer_tags, vectorizer_text = ModelBuilder.get_vectorizer_instance()
