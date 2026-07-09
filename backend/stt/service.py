"""
backend/stt/service.py
------------------------
Groq-hosted Whisper large-v3 for the demo, chosen over a self-hosted
or cold-start-prone alternative for a managed API with no cold start
and billing only for actual audio seconds.

This demo is English-only, so the language is hardcoded to "en" rather
than kept as unused branching for other languages.
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL   = "whisper-large-v3"


async def transcribe_audio(audio_bytes: bytes) -> dict:
    """
    Transcribe audio bytes to English text via Groq's managed Whisper large-v3.

    Args:
        audio_bytes: raw audio bytes from the browser (webm/opus).

    Returns:
        {
            "text":             str,   # transcribed text, empty on error
            "duration_seconds": float, # audio duration
            "error":            bool,  # True if transcription failed
        }

    No cold start: first call and hundredth call both complete in roughly
    0.5 to 2 seconds. Groq bills a 10-second minimum per request.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        logger.error("GROQ_API_KEY environment variable is not set")
        return {"text": "", "duration_seconds": 0.0, "error": True}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                GROQ_STT_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                data={
                    "model":           GROQ_MODEL,
                    "language":        "en",
                    "response_format": "verbose_json",
                    "temperature":     "0",
                },
                files={"file": ("recording.webm", audio_bytes, "audio/webm")},
            )

        if response.status_code == 200:
            data = response.json()
            return {
                "text":              (data.get("text") or "").strip(),
                "duration_seconds":  round(float(data.get("duration", 0.0)), 2),
                "error":             False,
            }

        logger.error(
            "Groq STT error | status=%d | %s",
            response.status_code, response.text[:200],
        )
        return {"text": "", "duration_seconds": 0.0, "error": True}

    except Exception as e:
        logger.error("STT transcription failed: %s: %s", type(e).__name__, e)
        return {"text": "", "duration_seconds": 0.0, "error": True}
