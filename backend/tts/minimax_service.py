"""
backend/tts/minimax_service.py
--------------------------------
MiniMax Speech-02-HD integration for the demo. This demo has one
language (English), so the voice map is just persona x gender (6
voices total), with no language dimension to branch on.

Voice IDs below are the ones you picked from MiniMax's English voice
library, confirmed to exist on Speech-02-HD's English variants (US, UK,
Australian, Indian accents are all under the "English" umbrella).
"""

import asyncio
import logging
import os

import httpx

logger = logging.getLogger(__name__)

MINIMAX_TTS_URL = "https://api-uw.minimax.io/v1/t2a_v2"

# -----------------------------------------------------------------------------
# VOICE MAP: 6 combinations (persona x gender)
#
# gender and persona are session-scoped choices (equivalent to what
# production reads from the user profile): session["voice_gender"] and
# session["persona"]. They flow through api/chat.py -> tts/service.py ->
# here, never hardcoded.
#
# To swap any voice: browse minimax.io/audio, copy the Voice ID, replace
# the string below.
# -----------------------------------------------------------------------------

VOICE_MAP = {
    # (persona, gender) -> voice_id
    ("caregiver", "female"): "English_radiant_girl",
    ("caregiver", "male"):   "English_ReservedYoungMan",
    ("friend",    "female"): "English_CalmWoman",
    ("friend",    "male"):   "English_causual_narrator_vv1",
    ("assistant", "female"): "English_captivating_female1",
    ("assistant", "male"):   "English_magnetic_voiced_man",
}

# -----------------------------------------------------------------------------
# SPEED MAP: per-voice overrides, empty for now.
#
# I can't hear the six English voices from here, so this starts empty.
# If any voice sounds too fast or slow once you test it, add an entry
# here keyed by (persona, gender).
# -----------------------------------------------------------------------------

DEFAULT_SPEED = 0.9
SPEED_MAP: dict[tuple[str, str], float] = {}

# -----------------------------------------------------------------------------
# EMOTION MAP: same structure as production, persona drives tone.
# -----------------------------------------------------------------------------

EMOTION_MAP = {
    "caregiver": "happy",
    "friend":    "happy",
    "assistant": "neutral",
}

# All six voices are English variants, so language_boost is constant here
# rather than a per-language lookup table.
LANGUAGE_BOOST = "English"

# -----------------------------------------------------------------------------
# FALLBACK VOICES
#
# If a primary voice ID is ever renamed or retired by MiniMax, a fixed
# fallback voice ID from their system library is one option, but that
# requires verifying it works under the English language_boost first.
# Rather than guess, this demo falls back to one of its own six voices
# per gender: assistant/female and assistant/male, on the assumption a
# "neutral assistant" voice is the safest default tone if a specific
# persona voice ever breaks. Swap these once you've actually listened
# to all six and have a preference.
# -----------------------------------------------------------------------------

FALLBACK_VOICES = {
    "female": VOICE_MAP[("assistant", "female")],
    "male":   VOICE_MAP[("assistant", "male")],
}

DEFAULT_VOICE   = VOICE_MAP[("assistant", "female")]
DEFAULT_EMOTION = "neutral"


class TTSError(Exception):
    """Raised when MiniMax TTS synthesis fails after all retries."""
    pass


async def synthesize(text: str, persona: str, gender: str) -> bytes:
    """
    Synthesize text to speech using MiniMax TTS API.

    Args:
        text:    reply text to speak
        persona: "caregiver", "friend", or "assistant"
        gender:  "female" or "male"

    Returns:
        Raw MP3 bytes.

    Raises:
        TTSError on failure after all retries and fallback exhausted.

    Fallback behavior: if the primary voice returns a MiniMax application
    error (voice not found, voice unavailable), retries once with the
    FALLBACK_VOICES entry for that gender. Silent to the person using the
    demo, logged as a warning.
    """
    api_key  = os.getenv("MINIMAX_API_KEY")
    group_id = os.getenv("MINIMAX_GROUP_ID")
    if not api_key:
        raise TTSError("MINIMAX_API_KEY environment variable is not set")
    if not group_id:
        raise TTSError("MINIMAX_GROUP_ID environment variable is not set")

    pers = persona.lower()
    gen  = gender.lower()

    primary_voice  = VOICE_MAP.get((pers, gen), DEFAULT_VOICE)
    fallback_voice = FALLBACK_VOICES.get(gen, DEFAULT_VOICE)
    emotion        = EMOTION_MAP.get(pers, DEFAULT_EMOTION)
    speed          = SPEED_MAP.get((pers, gen), DEFAULT_SPEED)

    for attempt_voice, is_fallback in [(primary_voice, False), (fallback_voice, True)]:
        if is_fallback and attempt_voice == primary_voice:
            break  # fallback is the same voice, no point retrying it

        try:
            audio_bytes = await _call_minimax(
                api_key=api_key,
                group_id=group_id,
                text=text,
                voice_id=attempt_voice,
                emotion=emotion,
                speed=speed,
            )
            if is_fallback:
                logger.warning(
                    "TTS primary voice '%s' failed, served fallback '%s' | "
                    "persona=%s gender=%s. Update VOICE_MAP if this repeats.",
                    primary_voice, attempt_voice, pers, gen,
                )
            return audio_bytes

        except TTSError as e:
            if is_fallback:
                raise
            logger.warning(
                "TTS primary voice '%s' error: %s, trying fallback '%s'",
                primary_voice, e, fallback_voice,
            )
            continue

    raise TTSError("TTS synthesis failed: primary and fallback both exhausted")


async def _call_minimax(
    api_key: str,
    group_id: str,
    text: str,
    voice_id: str,
    emotion: str,
    speed: float,
) -> bytes:
    """
    Single MiniMax API call with retry on transient errors. Same
    request shape and retry/backoff behavior as production.
    """
    payload = {
        "model": "speech-02-hd",
        "text":  text,
        "voice_setting": {
            "voice_id": voice_id,
            "speed":    speed,
            "pitch":    -1,
            "emotion":  emotion,
        },
        "language_boost": LANGUAGE_BOOST,
        "format": "mp3",
    }

    url     = f"{MINIMAX_TTS_URL}?GroupId={group_id}"
    retries = 3
    backoff = 1.0

    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(retries):
            try:
                response = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                )

                if response.status_code == 200:
                    data        = response.json()
                    base_resp   = data.get("base_resp", {})
                    status_code = base_resp.get("status_code", -1)

                    if status_code == 0:
                        audio_hex = data.get("data", {}).get("audio", "")
                        if not audio_hex:
                            raise TTSError("MiniMax returned 200 and status 0 but audio is empty")
                        return bytes.fromhex(audio_hex)

                    raise TTSError(
                        f"MiniMax app error {status_code}: {base_resp.get('status_msg')}"
                    )

                if response.status_code in (429, 500, 503):
                    if attempt < retries - 1:
                        logger.warning(
                            "MiniMax HTTP %d on attempt %d/%d, retrying in %.0fs",
                            response.status_code, attempt + 1, retries, backoff,
                        )
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue

                raise TTSError(
                    f"MiniMax HTTP {response.status_code}: {response.text[:200]}"
                )

            except httpx.TimeoutException:
                if attempt < retries - 1:
                    logger.warning(
                        "MiniMax timeout on attempt %d/%d, retrying in %.0fs",
                        attempt + 1, retries, backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                raise TTSError("MiniMax TTS timed out after all retries")

    raise TTSError("MiniMax TTS failed after all retries")
