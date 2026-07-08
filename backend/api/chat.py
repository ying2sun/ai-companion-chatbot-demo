"""
backend/api/chat.py
---------------------
The demo's /chat endpoint. This is a new, much smaller implementation,
not a port of production's api/chat.py, reflecting everything we agreed
to cut: no JWT auth, no Supabase profile, no tier/premium features, no
context enrichment, no profile-intent detection, no conversation
summarization, no speech monitoring, no S3. What's left is the actual
engineering core: transcribe (if voice), check guardrails, build the
prompt, call the LLM, detect chips, synthesize speech, respond.

One intentional behavior change from production, flagged rather than
silently carried over:

  Production's guardrails.py docstring says the check exists to
  "intercept before the LLM generates anything unsafe," but the real
  chat.py runs the LLM call FIRST and only checks the guardrail
  afterward (checking user_message, discarding whatever the LLM already
  generated if it fires). That means a crisis message still pays for a
  full Gemini call whose output gets thrown away. This demo runs the
  guardrail check before the LLM call instead, and skips the LLM
  entirely when it fires. That's both cheaper (no wasted API call) and
  closer to what the original docstring actually describes.

Pipeline:
  0. Rate limit check (per-IP, then the global daily cap)
  1. Validate input
  2. Load or create the in-memory session; apply this request's
     persona / voice_gender / display_name (live, every call)
  3. Transcribe audio, if provided
  4. Guardrail check on the user's message
  5a. Triggered: skip the LLM, use the override text
  5b. Not triggered: build system prompt, call the LLM, clean the reply
  6. Detect phone / URL chips in the reply (skipped if guardrail fired)
  7. Record the turn in session history
  8. Synthesize speech for the reply
  9. Return
"""

from __future__ import annotations

import logging
import re
import time

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from core.limiter import check_daily_cap, limiter
from llm.client import call_llm
from llm.guardrails import check_guardrails
from llm.prompts import build_system_prompt
from sessions.store import SessionStore
from stt.service import transcribe_audio
from suggestions.chips import detect_chips
from tts.service import synthesize_reply

logger = logging.getLogger(__name__)
router = APIRouter()
session_store = SessionStore()

_ALLOWED_PERSONAS = {"caregiver", "assistant", "friend"}
_ALLOWED_GENDERS = {"female", "male"}

# Belt-and-suspenders cleanup in case the model ignores the no-markdown
# instruction in the system prompt. Deliberately minimal, this is a
# fresh, small implementation, not a port of production's
# llm/post_processor.py, which wasn't part of the files shared for this
# build. Strips bold/italic markers and collapses stray blank lines; it
# does not attempt to enforce the sentence-count limit programmatically,
# truncating a reply mid-thought would do more harm than an occasional
# over-long response.
_MARKDOWN_RE = re.compile(r'(\*\*|\*|__|_)')


def _clean_response(text: str) -> str:
    cleaned = _MARKDOWN_RE.sub("", text)
    cleaned = re.sub(r'\n{2,}', '\n', cleaned).strip()
    return cleaned


class SuggestionItem(BaseModel):
    id: str
    label: str
    action: str
    target: str
    metadata: dict = {}


class ChatResponse(BaseModel):
    session_id: str
    reply_text: str
    persona: str
    guardrail_triggered: bool
    guardrail_reason: str | None
    latency_ms: int
    web_search_performed: bool
    audio_base64: str | None
    audio_mime: str | None
    input_audio_duration: float | None
    message_id: str
    user_message_id: str
    suggestions: list[SuggestionItem]


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("20/10minutes")
async def chat(
    request: Request,
    session_id: str | None = Form(None),
    message: str | None = Form(None),
    persona: str = Form("assistant"),
    voice_gender: str = Form("female"),
    display_name: str | None = Form(None),
    audio: UploadFile | None = File(None),
):
    # ── Rate limiting ────────────────────────────────────────────────────
    # Per-IP limit is enforced by the decorator above. This is the second,
    # independent layer: a global daily cap across every session combined,
    # see core/limiter.py for why both exist.
    if not check_daily_cap():
        raise HTTPException(status_code=429, detail="daily_limit_reached")

    # ── Input validation ─────────────────────────────────────────────────
    if persona not in _ALLOWED_PERSONAS:
        raise HTTPException(
            status_code=422,
            detail=f"persona must be one of {_ALLOWED_PERSONAS}, got '{persona}'",
        )
    if voice_gender not in _ALLOWED_GENDERS:
        raise HTTPException(
            status_code=422,
            detail=f"voice_gender must be one of {_ALLOWED_GENDERS}, got '{voice_gender}'",
        )
    if audio is None and (not message or not message.strip()):
        raise HTTPException(
            status_code=422,
            detail="Either 'audio' or a non-empty 'message' must be provided",
        )

    # ── Session ───────────────────────────────────────────────────────────
    session = session_store.get_or_create(session_id)
    session_id = session["session_id"]

    # Applied fresh every request, same reasoning as production: settings
    # changes should take effect on the very next turn, not require a
    # new session.
    session["persona"] = persona
    session["voice_gender"] = voice_gender
    if display_name:
        session["display_name"] = display_name

    # ── STT ───────────────────────────────────────────────────────────────
    input_audio_duration: float | None = None
    user_message = message.strip() if message else None

    if audio is not None:
        audio_bytes = await audio.read()
        stt_result = await transcribe_audio(audio_bytes)

        if stt_result["error"] or not stt_result["text"].strip():
            logger.warning("STT failed or returned empty transcript | session=%s", session_id)
            raise HTTPException(status_code=422, detail="stt_failed")

        user_message = stt_result["text"]
        input_audio_duration = stt_result["duration_seconds"]
        logger.info(
            "STT complete | session=%s | duration=%.1fs | chars=%d",
            session_id, input_audio_duration, len(user_message),
        )

    # ── Guardrail check, BEFORE the LLM call (see module docstring) ───────
    guardrail_result = check_guardrails(user_message)

    web_search_performed = False
    suggestions_raw: list[dict] = []
    t0 = time.monotonic()

    if guardrail_result["triggered"]:
        reply_text = guardrail_result["override_response"]
        logger.info(
            "Guardrail triggered: %s | session=%s", guardrail_result["reason"], session_id
        )
        # No chip detection, no LLM call, matches production's rule that
        # suggestions are cleared on a guardrail hit.

    else:
        profile = {
            "persona": session["persona"],
            "display_name": session["display_name"],
            "conversation_count": session["conversation_count"],
        }
        system_prompt = build_system_prompt(profile)

        messages = session_store.build_messages_for_llm(
            system_prompt, session["history"], user_message
        )

        try:
            raw_reply, web_search_performed, _source_urls = await call_llm(messages)
        except Exception as exc:
            logger.error("LLM call failed | session=%s | error=%s", session_id, exc, exc_info=True)
            raise HTTPException(status_code=500, detail="llm_error") from exc

        reply_text = _clean_response(raw_reply)
        suggestions_raw = detect_chips(reply_text)

    latency_ms = int((time.monotonic() - t0) * 1000)

    # ── Record turn ──────────────────────────────────────────────────────
    user_message_id, message_id = session_store.append_turn(session_id, user_message, reply_text)

    # ── TTS ───────────────────────────────────────────────────────────────
    audio_base64, audio_mime = await synthesize_reply(
        text=reply_text,
        persona=session["persona"],
        gender=session["voice_gender"],
    )
    if audio_base64 is None:
        logger.warning("TTS returned None, audio will be absent from response | session=%s", session_id)

    return ChatResponse(
        session_id=session_id,
        reply_text=reply_text,
        persona=session["persona"],
        guardrail_triggered=guardrail_result["triggered"],
        guardrail_reason=guardrail_result["reason"],
        latency_ms=latency_ms,
        web_search_performed=web_search_performed,
        audio_base64=audio_base64,
        audio_mime=audio_mime,
        input_audio_duration=input_audio_duration,
        message_id=message_id,
        user_message_id=user_message_id,
        suggestions=[SuggestionItem(**s) for s in suggestions_raw],
    )
