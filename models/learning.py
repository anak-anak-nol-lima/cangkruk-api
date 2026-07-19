from pydantic import BaseModel


class LearningMaterialRequest(BaseModel):
    system_prompt: str
    max_tokens: int = 16384


class LearningMaterialItem(BaseModel):
    id: str
    level: int
    title: str
    body: str
