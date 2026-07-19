from fastapi import APIRouter, HTTPException

from models.learning import LearningMaterialItem, LearningMaterialRequest
from services import llm

router = APIRouter(prefix="/learning", tags=["learning"])


@router.post("/materials", response_model=list[LearningMaterialItem])
async def materials(request: LearningMaterialRequest) -> list[LearningMaterialItem]:
    try:
        items = await llm.learning_material(request.system_prompt, request.max_tokens)
    except llm.LLMUpstreamError as error:
        raise HTTPException(status_code=502, detail=str(error))
    return [LearningMaterialItem(**item) for item in items]
