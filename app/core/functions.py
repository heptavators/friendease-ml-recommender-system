import numpy as np
import tensorflow as tf

from typing import List
from app.schemas import User
from app.core.logs import logger
from sklearn.metrics.pairwise import linear_kernel
from app.builder.model import tfrs, vectorizer_tags, vectorizer_text
from app.builder.data import talents_df, talents_tags_tfidf, talents_text_tfidf


def get_recommendation_with_tfidf(
    user: User, num_recommendations: int = 100
) -> List[str]:
    """

    Args:
        user (User): User data

    Returns:
        List[str]: List of recommended talents using TfidfVectorizer
    """

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

        recommended_talents = [talents_df.loc[i, "talent_id"] for i in talent_indices][
            :num_recommendations
        ]

        return recommended_talents
    except Exception:
        logger.error(
            "Something is wrong when trying to get recommendation talents using Tfidf"
        )

        return []


def get_recommendation_with_tfrs(user: User) -> List[str]:
    """

    Args:
        user (User): User data

    Returns:
        List[str]: List of recommended talents using TFRS
    """

    user_tags = "|".join(user.tags)

    try:
        _, recommended_talents = tfrs(
            {
                "user_id": tf.constant([user.id], dtype=tf.string),
                "user_gender": tf.constant([user.gender], dtype=tf.string),
                "user_age": tf.constant([user.age], dtype=tf.int32),
                "user_location": tf.constant([user.location], dtype=tf.string),
                "user_tags": tf.constant([user_tags], dtype=tf.string),
                "user_preferences": tf.constant([user.preferences], dtype=tf.string),
            }
        )

        return np.array(recommended_talents[0], dtype=np.str_).tolist()
    except Exception:
        logger.error(
            "Something is wrong when trying to get recommendation talents using TFRS"
        )

        return []


def get_recommended_talents(user: User) -> List[str]:
    """

    Args:
        user (User): User data

    Returns:
        List[str]: List of all recommended talents
    """

    tfidf_talents = get_recommendation_with_tfidf(user)
    tfrs_talents = get_recommendation_with_tfrs(user)

    user_location = user.location.split(",")[0]

    recommended_talents = []

    talent_with_same_location = talents_df.loc[
        (talents_df["talent_id"].isin(tfidf_talents))
        & (talents_df["talent_location"].str.contains(user_location, case=False)),
        "talent_id",
    ].values.tolist()
    talent_with_different_location = talents_df.loc[
        (talents_df["talent_id"].isin(tfidf_talents))
        & ~(talents_df["talent_location"].str.contains(user_location, case=False)),
        "talent_id",
    ].values.tolist()

    recommended_talents.extend(talent_with_same_location)
    recommended_talents.extend(talent_with_different_location)
    recommended_talents.extend(tfrs_talents)

    return recommended_talents
