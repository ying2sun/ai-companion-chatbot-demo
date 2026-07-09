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

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3-flash-preview"

_MAX_RETRIES = 3
_RETRY_BASE_WAIT = 2  # seconds

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Google Search grounding tool. google_search_retrieval with
# DynamicRetrievalConfig is hard-deprecated for Gemini 2.0+, only
# google_search is supported. The system prompt (prompts.py) is the only
# lever that influences when the model decides to search.
_SEARCH_TOOL = types.Tool(google_search=types.GoogleSearch())

# Local MCP server (mcp_tools/units_server.py), launched as a subprocess
# over stdio each time the tool is actually invoked.
#
# This does NOT pass a live MCP session into the Gemini call, on
# purpose, an earlier version did (tools=[mcp_client.session], the
# pattern Google's own docs and FastMCP's docs both describe), and it
# broke in production with "cannot pickle '_asyncio.Future' object".
# Root cause, confirmed by reading the installed SDK's own source, not
# guessed at: generate_content() unconditionally deep-copies the whole
# config as its very first line, before its own MCP-session-handling
# code ever runs to strip that session back out. That handling code
# exists and looks correct, it's just unreachable, since the crash
# happens one line earlier. So a live session can never safely sit
# inside a config passed to this SDK version at all.
#
# The fix: declare the tool's schema statically (safe to copy, it's
# just data) and handle the call-and-respond round trip by hand,
# mirroring the exact pattern the SDK's own automatic function calling
# uses internally (types.Part.from_function_response, a role='user'
# Content carrying the result), just without the buggy live-session
# path.
_UNITS_MCP_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "mcp_tools", "units_server.py")

_UNITS_FUNCTION_DECLARATION = types.FunctionDeclaration(
    name="convert_units",
    description=(
        "Convert a measurement to its natural counterpart unit. "
        "Supported from_unit values: C, F, kg, lb, km, miles, cm. "
        "Temperature converts to the other temperature scale. Weight "
        "and distance convert to their common alternate unit. Height "
        "(cm) converts to feet and inches."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "value": {"type": "number"},
            "from_unit": {"type": "string"},
        },
        "required": ["value", "from_unit"],
        "additionalProperties": False,
    },
)
_UNITS_TOOL = types.Tool(function_declarations=[_UNITS_FUNCTION_DECLARATION])


async def _execute_units_tool(args: dict) -> dict:
    """Run the actual conversion through the real MCP protocol, a
    fresh short-lived connection per call, same reasoning as before:
    this app can serve concurrent requests, so connections aren't
    shared across calls."""
    async with MCPClient(_UNITS_MCP_SCRIPT) as client:
        result = await client.call_tool("convert_units", args)
        return result.data

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

    config = types.GenerateContentConfig(
        system_instruction=system_instruction or None,
        max_output_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        tools=[_SEARCH_TOOL, _UNITS_TOOL],
        # Gemini treats google_search (a server-side built-in tool) and
        # function_declarations (client-side function calling)
        # differently, and refuses to combine them in one request
        # unless told to explicitly. Confirmed by the API's own error
        # message, not guessed at: "Please enable
        # tool_config.include_server_side_tool_invocations to use
        # Built-in tools with Function calling."
        tool_config=types.ToolConfig(include_server_side_tool_invocations=True),
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    response = await _client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    # Manual function-calling round trip. One call max, this handles a
    # single tool invocation per turn, not chained multi-step tool use,
    # a deliberate scope limit for what this demo actually needs.
    function_call_part = None
    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.function_call:
                function_call_part = part.function_call
                break

    if function_call_part is not None and function_call_part.name == "convert_units":
        try:
            tool_result = await _execute_units_tool(dict(function_call_part.args or {}))
            func_response = {"result": tool_result}
        except Exception as exc:
            logger.warning("Units MCP tool call failed | args=%s | error=%s", function_call_part.args, exc)
            func_response = {"error": str(exc)}

        func_response_part = types.Part.from_function_response(
            name=function_call_part.name, response=func_response
        )
        contents = list(contents)
        contents.append(response.candidates[0].content)
        contents.append(types.Content(role="user", parts=[func_response_part]))

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
