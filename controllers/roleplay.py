from fastapi import APIRouter, HTTPException

from models.roleplay import (
    FeedbackRequest,
    FeedbackResponse,
    ReplyRequest,
    ReplyResponse,
)
from services import llm

router = APIRouter(prefix="/roleplay", tags=["roleplay"])

EVALUATOR_PROMPT = """\
Kamu adalah senior barista yang menilai latihan komunikasi barista junior. \
Nilai berdasarkan: sapaan, konfirmasi pesanan, pengetahuan menu, dan penutup interaksi. \
Jawab dalam bahasa Indonesia, panggil barista dengan "kamu", dan pakai format PERSIS seperti ini:
SUMMARY: <2-3 kalimat penilaian keseluruhan>
FEEDBACK: <2-3 kalimat saran konkret beserta contoh kalimat yang bisa langsung dipakai>"""


@router.post("/reply", response_model=ReplyResponse)
async def reply(request: ReplyRequest) -> ReplyResponse:
    if not request.messages:
        raise HTTPException(status_code=422, detail="messages tidak boleh kosong")
    try:
        text = await llm.chat(
            request.system_prompt,
            [turn.model_dump() for turn in request.messages],
            max_tokens=512,
        )
    except llm.LLMUpstreamError as error:
        raise HTTPException(status_code=502, detail=str(error))
    return ReplyResponse(reply=text)


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    prompt = (
        f"Berikut transkrip latihannya:\n\n{request.transcript}\n\n"
        "Berikan penilaianmu sesuai format."
    )
    try:
        raw = await llm.chat(
            EVALUATOR_PROMPT, [{"role": "barista", "text": prompt}], max_tokens=2048
        )
    except llm.LLMUpstreamError as error:
        raise HTTPException(status_code=502, detail=str(error))
    return FeedbackResponse(**parse_feedback(raw))


def _strip_label(text: str, label: str) -> str:
    idx = text.upper().find(label)
    return text[idx + len(label):] if idx != -1 else text


def parse_feedback(raw: str) -> dict:
    text = raw.strip()
    marker = text.upper().find("FEEDBACK:")
    if marker == -1:
        # tanpa marker FEEDBACK: tetap buang prefix SUMMARY: biar tidak bocor
        return {"summary": _strip_label(text, "SUMMARY:").strip(), "feedback": ""}
    summary = _strip_label(text[:marker], "SUMMARY:")
    return {
        "summary": summary.strip(),
        "feedback": text[marker + len("FEEDBACK:"):].strip(),
    }
