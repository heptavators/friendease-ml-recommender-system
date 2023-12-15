from pydantic import BaseModel, Field


class User(BaseModel):
    """Base payload for the user data"""

    id: str = Field(description="User id (uuid4)")
    name: str = Field(description="User full name")
    gender: str = Field(description="User gender (P | L)")
    birth_date: str = Field(description="User birth_date (dd/mm/YYYY)")
    age: int = Field(description="User age")
    location: str = Field(description="User location")
    tags: list[str] = Field(description="List of user tags")
    preferences: str = Field(description="User preferences for searching talent")


class Talent(BaseModel):
    """Base payload for the talent data"""

    id: str = Field(description="Talent id (uuid4)")


class ListTalent(BaseModel):
    """Base payload for the recommended talents"""

    data: list[Talent] = Field(description="List of recommended talents")
    message: str = Field(description="Message for the response")
