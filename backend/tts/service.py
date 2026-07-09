"""
backend/tts/service.py
------------------------
Thin wrapper between api/chat.py and minimax_service.py. Only this
file is imported by /chat; minimax_service.py is never imported
directly outside this module.

This demo has no persistent storage, audio only needs to survive the
current HTTP response, so it skips uploading anywhere and returns the
raw MP3 bytes (base64-encoded for JSON transport) directly. The
frontend builds a data URL or Blob from it and plays it immediately.
A production system with multi-session history would instead persist
the file somewhere durable and return a URL, since audio needs to
survive across sessions there.
"""

import base64
import logging

from tts.minimax_service import TTSError, synthesize

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
