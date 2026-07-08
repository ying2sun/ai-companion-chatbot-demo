"""
backend/tts/service.py
------------------------
Thin wrapper between api/chat.py and google_service.py. Same separation
of concerns as production: only this file is imported by /chat,
google_service.py is never imported directly outside this module.

What's different from production, and why:
  Production uploads the synthesized MP3 to S3 and returns a 7-day
  pre-signed URL plus the permanent S3 key, because audio needs to
  survive across sessions and be replayable after the URL expires. This
  demo has no persistent storage at all, audio only needs to survive the
  current HTTP response. So this version skips S3 entirely and returns
  the raw MP3 bytes (base64-encoded for JSON transport) directly. The
  frontend builds a data URL or Blob from it and plays it immediately,
  the same approach production itself prototyped and proved out as
  "Option A" before reverting it for unrelated Flutter-coordination
  reasons, not because it didn't work.
"""

import base64
import logging

from tts.google_service import TTSError, synthesize

logger = logging.getLogger(__name__)


async def synthesize_reply(
    text: str,
    persona: str = "assistant",
    gender: str = "female",
) -> tuple[str | None, str | None]:
    """
    Synthesize text to speech and return it ready for inline JSON transport.

    Returns:
        (audio_base64, mime_type) on success, both str.
        (None, None) on any failure.

    Error behavior mirrors production: TTS is an enhancement, never a
    blocker. reply_text is always returned by /chat regardless of
    whether this succeeds.
    """
    try:
        audio_bytes = await synthesize(text, persona, gender)
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        logger.info(
            "TTS complete | persona=%s | gender=%s | chars=%d | bytes=%d",
            persona, gender, len(text), len(audio_bytes),
        )
        return audio_b64, "audio/mpeg"

    except TTSError as e:
        logger.error("TTS synthesis failed | error=%s", e)
        return None, None
    except Exception as e:
        logger.error("TTS unexpected error | error=%s", e, exc_info=True)
        return None, None
