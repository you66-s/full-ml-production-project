from pydantic import BaseModel

class UserPayload(BaseModel):
    user_id: str