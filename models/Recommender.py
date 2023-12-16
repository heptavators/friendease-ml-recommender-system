from . import schemas
from common import functions
from logs.logger import logger


def get_list_talents(user: schemas.User) -> schemas.ListTalent:
    recommended_talents = functions.get_recommended_talents(user)
    data = [schemas.Talent(id=id) for id in recommended_talents]
    response = schemas.ListTalent(
        data=data, message="Successfully getting recommendation"
    )

    return response
