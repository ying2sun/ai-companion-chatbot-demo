"""
backend/llm/client.py
-----------------------
Gemini API wiring. This file is almost entirely reused from production
as-is, not rewritten, because it's generic API integration code (message
format conversion, retry logic, grounding metadata extraction) with no
Chinese-specific content in it. The only changes: comments that
referenced production's 2-chip suggestion limit are generalized, since
this demo's chip system is smaller but still caps around the same
number.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from dotenv import load_dotenv
from fastmcp import Client as MCPClient

from core.env import clean_env_secret

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3-flash-preview"

_MAX_RETRIES = 3
_RETRY_BASE_WAIT = 2  # seconds

_client = genai.Client(api_key=clean_env_secret(os.getenv("GEMINI_API_KEY")))

# Google Search grounding tool. google_search_retrieval with
# DynamicRetrievalConfig is hard-deprecated for Gemini 2.0+, only
# google_search is supported. The system prompt (prompts.py) is the only
# lever that influences when the model decides to search.
_SEARCH_TOOL = types.Tool(google_search=types.GoogleSearch())

# Local MCP server (mcp_tools/units_server.py), launched as a subprocess
# over stdio for the duration of each call. The google-genai SDK's MCP
# support is currently marked experimental by Google, worth knowing if
# this ever misbehaves, it's a newer code path than the rest of this
# file. A fresh Client is constructed per call rather than sharing one
# instance across requests, since this app can serve concurrent
# requests and a shared client's connection isn't meant to be entered
# from multiple calls at once. A production version would want a
# persistent connection pool instead of paying subprocess startup cost
# on every single turn, a reasonable next step, not done here.
_UNITS_MCP_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "mcp_tools", "units_server.py")

# Max source URLs to extract from grounding_metadata and return to
# chat.py. Surfaced as chips only when the person explicitly asks for a
# source.
_MAX_SOURCE_URLS = 2


async def call_llm(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.8,
) -> tuple[str, bool, list[str]]:
    """
    Send a conversation history to Gemini and return the reply.

    Accepts messages in OpenAI format, conversion to google-genai format
    happens internally in _call_once. Callers (chat.py, sessions/) don't
    need to change.

    Returns: (reply_text, web_search_performed, source_urls)
      - reply_text: cleaned string ready for post-processing
      - web_search_performed: True if Gemini used Google Search grounding
      - source_urls: up to 2 source URLs from grounding_metadata, empty
        if no search was performed or no URLs were returned

    Raises: google.genai.errors.APIError on non-2xx after all retries.
            ValueError if the model returns empty content.
    """
    last_exc = None

    for attempt in range(_MAX_RETRIES):
        try:
            return await _call_once(messages, model, max_tokens, temperature)
        except genai_errors.APIError as exc:
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            is_rate_limit = status == 429 or "429" in str(exc)

            if is_rate_limit and attempt < _MAX_RETRIES - 1:
                wait = _RETRY_BASE_WAIT ** (attempt + 1)
                logger.warning(
                    "Gemini 429 rate limit, retrying in %ds (attempt %d/%d)",
                    wait, attempt + 1, _MAX_RETRIES,
                )
                await asyncio.sleep(wait)
                last_exc = exc
            else:
                raise

    raise last_exc


def _convert_messages(messages: list[dict]) -> tuple[str, list[types.Content]]:
    """Convert OpenAI-format messages to google-genai Content objects."""
    system_instruction = ""
    contents = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            system_instruction = content if isinstance(content, str) else ""
            continue

        genai_role = "model" if role == "assistant" else "user"
        parts = _convert_content_to_parts(content)

        if parts:
            contents.append(types.Content(role=genai_role, parts=parts))

    return system_instruction, contents


def _convert_content_to_parts(content) -> list[types.Part]:
    """Convert a single message's content field to a list of google-genai Parts."""
    if isinstance(content, str):
        return [types.Part(text=content)]

    if isinstance(content, list):
        parts = []
        for item in content:
            item_type = item.get("type", "")

            if item_type == "text":
                parts.append(types.Part(text=item.get("text", "")))

            elif item_type == "image_url":
                url = item.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    try:
                        header, b64data = url.split(",", 1)
                        mime_type = header.split(":")[1].split(";")[0]
                        image_bytes = base64.b64decode(b64data)
                        parts.append(types.Part(
                            inline_data=types.Blob(
                                mime_type=mime_type,
                                data=image_bytes,
                            )
                        ))
                    except Exception as e:
                        logger.warning("Failed to decode image in message: %s", e)
                else:
                    logger.warning("Skipping non-base64 image URL: %s", url[:60])

        return parts

    return [types.Part(text=str(content))]


async def _call_once(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, bool, list[str]]:
    """Single attempt at the Gemini API. No retry logic here."""
    system_instruction, contents = _convert_messages(messages)

    units_mcp_client = MCPClient(_UNITS_MCP_SCRIPT)

    async with units_mcp_client:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction or None,
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            tools=[_SEARCH_TOOL, units_mcp_client.session],
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        response = await _client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

    finish_reason = "unknown"
    if response.candidates:
        raw_reason = response.candidates[0].finish_reason
        finish_reason = raw_reason.name if hasattr(raw_reason, "name") else str(raw_reason)

    # Primary extraction via response.text. Can silently return empty
    # when grounding is active (non-text grounding parts mixed in).
    # Falls back to manual parts iteration below.
    try:
        content = (response.text or "").strip()
    except Exception:
        content = ""

    if not content and response.candidates:
        try:
            parts = response.candidates[0].content.parts or []
            content = "".join(
                p.text for p in parts if hasattr(p, "text") and p.text
            ).strip()
            if content:
                logger.info("Used parts fallback for content extraction (grounding active)")
        except Exception as e:
            logger.warning("Parts fallback also failed: %s", e)
            content = ""

    logger.info(
        "Gemini response | finish_reason=%s | output_chars=%d | model=%s",
        finish_reason, len(content), model,
    )

    if finish_reason == "SAFETY":
        logger.warning("Gemini safety filter triggered | model=%s", model)
        raise ValueError("Gemini safety block: finish_reason=SAFETY")

    if finish_reason == "MAX_TOKENS":
        logger.warning(
            "Gemini hit max_output_tokens (%d), response may be truncated | output_chars=%d",
            max_tokens, len(content),
        )

    if not content:
        logger.error(
            "LLM returned empty content | model=%s | finish_reason=%s",
            model, finish_reason,
        )
        raise ValueError("LLM returned empty content")

    web_search_performed = False
    source_urls: list[str] = []
    try:
        if response.candidates and response.candidates[0].grounding_metadata:
            web_search_performed = True
            chunks = response.candidates[0].grounding_metadata.grounding_chunks or []
            for chunk in chunks:
                if len(source_urls) >= _MAX_SOURCE_URLS:
                    break
                if hasattr(chunk, "web") and chunk.web and chunk.web.uri:
                    source_urls.append(chunk.web.uri)
    except Exception:
        pass  # never crash the pipeline on metadata parsing

    if web_search_performed:
        logger.info("Gemini used Google Search grounding | sources=%d", len(source_urls))

    return content, web_search_performed, source_urls
