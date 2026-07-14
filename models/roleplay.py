from pydantic import BaseModel


class ChatTurn(BaseModel):
    role: str  
    text: str


class ReplyRequest(BaseModel):
    system_prompt: str
    messages: list[ChatTurn]


class ReplyResponse(BaseModel):
    reply: str


class FeedbackRequest(BaseModel):
    transcript: str


class FeedbackResponse(BaseModel):
    summary: str
    feedback: str
