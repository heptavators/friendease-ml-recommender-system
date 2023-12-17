from .schemas import User, Talent, ListTalent
from common import functions
from logs.logger import logger


def get_list_talents(user: User) -> ListTalent:
    recommended_talents = functions.get_recommended_talents(user)
    data = [Talent(id=id) for id in recommended_talents]
    response = ListTalent(data=data, message="Successfully getting recommendation")

    return response
