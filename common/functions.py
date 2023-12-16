import os
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
        talents = pd.read_csv(os.path.join(os.getcwd(), "data", "talents.csv"))

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


def get_recommendation_with_tfidf(
    user: schemas.User, num_recommendations: int = 100
) -> list[str]:
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

    try:
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
        recommended_talents = [talents_df.loc[i, "talent_id"] for i in talent_indices][
            :num_recommendations
        ]
    except Exception:
        logger.error(
            "Something is wrong when trying to get recommendation talents using Tfidf"
        )

    return recommended_talents


def get_recommendation_with_tfrs(user: schemas.User) -> list[str]:
    """

    Args:
        user (schemas.User): User data

    Returns:
        list[str]: List of recommended talents using TFRS
    """
    index = builder.build_model()
    user_tags = "|".join(user.tags)

    try:
        _, recommended_talents = index(
            {
                "user_id": np.array([user.id]),
                "user_gender": np.array([user.gender]),
                "user_age": np.array([user.age]),
                "user_location": np.array([user.location]),
                "user_tags": np.array([user_tags]),
                "user_preferences": np.array([user.preferences]),
            }
        )
    except Exception:
        logger.error(
            "Something is wrong when trying to get recommendation talents using TFRS"
        )

    return np.array(recommended_talents[0], dtype=np.str_).tolist()


def get_recommended_talents(user: schemas.User) -> list[str]:
    """

    Args:
        user (schemas.User): User data

    Returns:
        list[str]: List of all recommended talents
    """
    talents = get_talents()
    tfidf_talents = get_recommendation_with_tfidf(user)
    tfrs_talents = get_recommendation_with_tfrs(user)

    user_location = user.location.split(",")[0]

    recommended_talents = []

    talent_with_same_location = talents.loc[
        (talents["talent_id"].isin(tfidf_talents))
        & (talents["talent_location"].str.contains(user_location, case=False)),
        "talent_id",
    ].values.tolist()
    talent_with_different_location = talents.loc[
        (talents["talent_id"].isin(tfidf_talents))
        & ~(talents["talent_location"].str.contains(user_location, case=False)),
        "talent_id",
    ].values.tolist()

    recommended_talents.extend(talent_with_same_location)
    recommended_talents.extend(talent_with_different_location)
    recommended_talents.extend(tfrs_talents)

    return recommended_talents
