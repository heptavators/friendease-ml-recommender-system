import numpy as np

from logs.logger import logger
from models.schemas import User
from sklearn.metrics.pairwise import linear_kernel
from models.builder import index, vectorizer_tags, vectorizer_text
from common.data import talents_df, talents_tags_tfidf, talents_text_tfidf


def get_recommendation_with_tfidf(
    user: User, num_recommendations: int = 100
) -> list[str]:
    """

    Args:
        user (User): User data

    Returns:
        list[str]: List of recommended talents using TfidfVectorizer
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

        logger.debug(recommended_talents[:5])

        return recommended_talents
    except Exception:
        logger.error(
            "Something is wrong when trying to get recommendation talents using Tfidf"
        )

        return []


def get_recommendation_with_tfrs(user: User) -> list[str]:
    """

    Args:
        user (User): User data

    Returns:
        list[str]: List of recommended talents using TFRS
    """

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

        logger.debug(recommended_talents[0][:5])

        return np.array(recommended_talents[0], dtype=np.str_).tolist()
    except Exception:
        logger.error(
            "Something is wrong when trying to get recommendation talents using TFRS"
        )

        return []


def get_recommended_talents(user: User) -> list[str]:
    """

    Args:
        user (User): User data

    Returns:
        list[str]: List of all recommended talents
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
