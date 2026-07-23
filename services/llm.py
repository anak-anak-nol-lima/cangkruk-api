import asyncio
import json
import os
import re
import uuid

import httpx

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# dicoba berurutan, kalau model utama penuh (503) atau ditutup (404),
# geser ke model berikutnya, model lite biasanya masih longgar kapasitasnya
FALLBACK_MODELS = ["gemini-3.1-flash-lite", "gemini-2.5-flash-lite"]


class LLMUpstreamError(Exception):
    pass


async def chat(system_prompt: str, turns: list[dict], max_tokens: int = 256) -> str:
    """Kirim percakapan ke Gemini, balikin teks jawabannya."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMUpstreamError("GEMINI_API_KEY belum di-set di environment server")

    primary = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
    models = [primary] + [m for m in FALLBACK_MODELS if m != primary]

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

    last_error = "tidak ada model yang bisa dicoba"
    async with httpx.AsyncClient(timeout=60) as client:
        for model in models:
            for attempt in range(2):
                response = await client.post(
                    GEMINI_URL.format(model=model),
                    params={"key": api_key},
                    json=payload,
                )

                if response.status_code == 200:
                    data = response.json()
                    try:
                        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    except (KeyError, IndexError):
                        raise LLMUpstreamError(
                            f"Bentuk respons Gemini tidak terduga: {str(data)[:200]}"
                        )

                last_error = f"{model} balas {response.status_code}: {response.text[:150]}"

                if response.status_code in (429, 503):
                    # penuh/kena limit: antre sebentar lalu coba sekali lagi,
                    # habis itu nyerah dan pindah ke model berikutnya
                    await asyncio.sleep(1.5)
                    continue
                # 404/400 dll: retry tidak akan menolong, langsung model berikutnya
                break

    raise LLMUpstreamError(f"Semua model sedang penuh/gagal. Terakhir: {last_error}")


async def learning_material(system_prompt: str, max_tokens: int = 16384) -> list[dict]:
    """Minta Gemini menyusun kurikulum terstruktur, balikin list materi.

    Tiap materi: {id, level, title, body}. Level dibatasi 1..4, id di-assign
    di sini sebagai UUID supaya unik dan tidak bergantung pada model.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMUpstreamError("GEMINI_API_KEY belum di-set di environment server")

    primary = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
    models = [primary] + [m for m in FALLBACK_MODELS if m != primary]

    # kontrak keluaran kita, dikirim sebagai giliran user supaya system_prompt
    # milik pemanggil tetap utuh apa adanya
    contract = (
        "Susun materi pembelajaran sebagai array JSON. Setiap item punya level, "
        "title, dan body. Level: 1=easy, 2=medium, 3=hard, 4=specialist (maksimum 4). "
        "Materi level 1 (easy) biasanya banyak, jadi pecah jadi beberapa item terpisah "
        "yang masing-masing fokus ke satu konsep, jangan digabung jadi satu item besar."
    )

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": contract}]}],
        "generationConfig": {
            "temperature": 0.8,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "level": {"type": "INTEGER"},
                        "title": {"type": "STRING"},
                        "body": {"type": "STRING"},
                    },
                    "required": ["level", "title", "body"],
                    "propertyOrdering": ["level", "title", "body"],
                },
            },
        },
    }

    last_error = "tidak ada model yang bisa dicoba"
    async with httpx.AsyncClient(timeout=60) as client:
        for model in models:
            for attempt in range(2):
                response = await client.post(
                    GEMINI_URL.format(model=model),
                    params={"key": api_key},
                    json=payload,
                )

                if response.status_code == 200:
                    data = response.json()
                    try:
                        raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    except (KeyError, IndexError):
                        raise LLMUpstreamError(
                            f"Bentuk respons Gemini tidak terduga: {str(data)[:200]}"
                        )

                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        # biasanya kepotong karena kehabisan maxOutputTokens
                        # (token "mikir" Gemini ikut makan jatah), naikkan max_tokens
                        finish = data["candidates"][0].get("finishReason", "?")
                        raise LLMUpstreamError(
                            f"Respons Gemini bukan JSON valid (finishReason={finish}, "
                            f"kemungkinan kepotong): {raw[-200:]}"
                        )

                    if not isinstance(parsed, list):
                        raise LLMUpstreamError(f"Respons Gemini bukan array: {str(parsed)[:200]}")

                    materials = []
                    for entry in parsed:
                        if not isinstance(entry, dict):
                            continue
                        title = entry.get("title")
                        body = entry.get("body")
                        if not title or not body:
                            continue
                        # buang heading markdown (#, ##, ###, ####), teksnya tetap
                        body = "\n".join(
                            re.sub(r"^\s*#{1,4}\s*", "", line)
                            for line in body.splitlines()
                        )
                        level = entry.get("level", 1)
                        try:
                            level = int(level)
                        except (TypeError, ValueError):
                            level = 1
                        # level maksimum 4: 1 easy, 2 medium, 3 hard, 4 specialist
                        level = max(1, min(4, level))
                        materials.append(
                            {
                                "id": str(uuid.uuid4()),
                                "level": level,
                                "title": title,
                                "body": body,
                            }
                        )

                    materials.sort(key=lambda m: m["level"])
                    return materials

                last_error = f"{model} balas {response.status_code}: {response.text[:150]}"

                if response.status_code in (429, 503):
                    # penuh/kena limit: antre sebentar lalu coba sekali lagi,
                    # habis itu nyerah dan pindah ke model berikutnya
                    await asyncio.sleep(1.5)
                    continue
                # 404/400 dll: retry tidak akan menolong, langsung model berikutnya
                break

    raise LLMUpstreamError(f"Semua model sedang penuh/gagal. Terakhir: {last_error}")
