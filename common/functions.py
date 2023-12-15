import pandas as pd
import numpy as np

from logs.logger import logger
from models import builder, schemas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def get_talents() -> pd.DataFrame:
    """

    Returns:
        pd.DataFrame: List of existed talents
    """

    global talents

    if not "talents" in globals():
        talents = pd.read_csv("../data/talents.csv")

    return talents


def get_talents_tfidf_matrix(
    vectorizer_tags: TfidfVectorizer, vectorizer_text: TfidfVectorizer
) -> tuple[np.ndarray]:
    """

    Returns:
        np.ndarray: Tuples of tfidf matrix
    """
    global talents_tags_tfidf, talents_text_tfidf

    talents_df = get_talents()

    if not "talents_tags_tfidf" in globals():
        talents_tags_tfidf = vectorizer_tags.transform(talents_df["talent_tags"])

    if not "talents_text_tfidf" in globals():
        talents_text_tfidf = vectorizer_text.transform(talents_df["talent_description"])

    return talents_tags_tfidf, talents_text_tfidf


def get_recommendation_with_tfidf(user: schemas.User) -> list[str]:
    """

    Args:
        user (schemas.User): User data

    Returns:
        list[str]: List of recommended talents using TfidfVectorizer
    """

    vectorizer_tags, vectorizer_text = builder.build_vectorizer()
    talents_tags_tfidf, talents_text_tfidf = get_talents_tfidf_matrix(
        vectorizer_tags, vectorizer_text
    )

    user_tags = "|".join(user.tags)
    user_preferences = user.preferences

    user_tags_tfidf = vectorizer_tags.transform([user_tags])
    user_text_tfidf = vectorizer_text.transform([user_preferences])

    cosine_similarities_tags = linear_kernel(user_tags_tfidf, talents_tags_tfidf)
    cosine_similarities_text = linear_kernel(user_text_tfidf, talents_text_tfidf)

    cosine_similarities_combined = (
        0.5 * cosine_similarities_tags + 0.5 * cosine_similarities_text
    )

    # Get the indices of talents sorted by similarity
    talent_indices = cosine_similarities_combined.argsort()[0][::-1]

    talents_df = get_talents()
    num_recommendations = 100
    recommended_talents = [talents_df.loc[i, "talent_id"] for i in talent_indices][
        :num_recommendations
    ]

    return recommended_talents


def get_recommendation_with_tfrs(user: schemas.User) -> list[str]:
    """

    Args:
        user (schemas.User): User data

    Returns:
        list[str]: List of recommended talents using TFRS
    """
    index = builder.build_model()

    _, recommended_talents = index(
        {
            "user_id": np.array([user.id]),
            "user_gender": np.array([user.gender]),
            "user_age": np.array([user.tags]),
            "user_location": np.array([user.location]),
            "user_tags": np.array([user.tags]),
            "user_preferences": np.array([user.preferences]),
        },
        k=100,
    )

    return recommended_talents
