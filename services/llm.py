import asyncio
import os

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

    primary = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
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
