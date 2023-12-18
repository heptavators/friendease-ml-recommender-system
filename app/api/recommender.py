from app.core import functions
from app.schemas import User, Talent, ListTalent


def get_list_talents(user: User) -> ListTalent:
    recommended_talents = functions.get_recommended_talents(user)
    data = [Talent(id=id) for id in recommended_talents]
    response = ListTalent(data=data, message="Successfully getting recommendation")

    return response
