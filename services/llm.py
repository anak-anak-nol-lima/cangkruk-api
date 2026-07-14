import os

import httpx

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class LLMUpstreamError(Exception):
    pass


async def chat(system_prompt: str, turns: list[dict], max_tokens: int = 256) -> str:
    """Kirim percakapan ke Gemini, balikin teks jawabannya."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMUpstreamError("GEMINI_API_KEY belum di-set di environment server")

    # gemini-2.5-flash ditutup untuk user baru (Jul 2026) — 3.5-flash penggantinya
    model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {
                "role": "user" if turn["role"] == "barista" else "model",
                "parts": [{"text": turn["text"]}],
            }
            for turn in turns
        ],
        "generationConfig": {"temperature": 0.8, "maxOutputTokens": max_tokens},
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            GEMINI_URL.format(model=model),
            params={"key": api_key},
            json=payload,
        )

    if response.status_code != 200:
        raise LLMUpstreamError(f"Gemini balas {response.status_code}: {response.text[:200]}")

    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        raise LLMUpstreamError(f"Bentuk respons Gemini tidak terduga: {str(data)[:200]}")
