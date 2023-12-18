import pandas as pd
import numpy as np

from app.core.config import settings
from .model import vectorizer_tags, vectorizer_text


class DataBuilder:
    _talents_df = None
    _talents_tags_tfidf = None
    _talents_text_tfidf = None

    @staticmethod
    def get_talents_df():
        if DataBuilder._talents_df is None:
            DataBuilder._talents_df = pd.read_csv(settings.CONFIG["data"]["talents"])
        return DataBuilder._talents_df

    @staticmethod
    def get_talents_tfidf_matrix():
        talents_df = DataBuilder.get_talents_df()

        if DataBuilder._talents_tags_tfidf is None:
            DataBuilder._talents_tags_tfidf = vectorizer_tags.transform(
                talents_df["talent_tags"]
            )

        if DataBuilder._talents_text_tfidf is None:
            DataBuilder._talents_text_tfidf = vectorizer_text.transform(
                talents_df["talent_description"]
            )

        return (
            DataBuilder._talents_tags_tfidf,
            DataBuilder._talents_text_tfidf,
        )


talents_df = DataBuilder.get_talents_df()
talents_tags_tfidf, talents_text_tfidf = DataBuilder.get_talents_tfidf_matrix()
