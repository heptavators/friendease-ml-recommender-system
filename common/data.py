import pandas as pd
import numpy as np

from .config import config
from models.builder import vectorizer_tags, vectorizer_text


def get_talents() -> pd.DataFrame:
    """

    Returns:
        pd.DataFrame: List of existed talents
    """

    if "talents" not in get_talents.__dict__:
        get_talents.talents = pd.read_csv(config["data"]["talents"])

    return get_talents.talents


talents_df = get_talents()


def get_talents_tfidf_matrix() -> tuple[np.ndarray]:
    """

    Returns:
        np.ndarray: Tuples of tfidf matrix
    """

    if "talents_tags_tidf" not in get_talents_tfidf_matrix.__dict__:
        talents_tags_tfidf = vectorizer_tags.transform(talents_df["talent_tags"])

    if "talents_text_tfidf" not in get_talents_tfidf_matrix.__dict__:
        talents_text_tfidf = vectorizer_text.transform(talents_df["talent_description"])

    return talents_tags_tfidf, talents_text_tfidf


talents_tags_tfidf, talents_text_tfidf = get_talents_tfidf_matrix()
