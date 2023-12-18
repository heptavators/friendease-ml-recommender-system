from typing import List
from pydantic import BaseModel, Field


class Talent(BaseModel):
    """Base payload for the talent data"""

    id: str = Field(description="Talent id (uuid4)")


class ListTalent(BaseModel):
    """Base payload for the recommended talents"""

    data: List[Talent] = Field(description="List of recommended talents")
    message: str = Field(description="Message for the response")
