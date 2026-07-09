"""
backend/sessions/store.py
---------------------------
In-memory session store. A production system with persistent identity
would back this with a database and per-row IDs meant to survive
across visits; this demo has no database and no persistent identity,
a session exists only for the lifetime of one browser tab.

Interface is shaped to match what api/chat.py needs: get_or_create(),
build_messages_for_llm(), append_turn().

TTL eviction: since there's no background worker in this lightweight
build, stale sessions are swept lazily, on every call, rather than on a
schedule. At demo traffic levels a full dict scan on each request costs
nothing measurable, and it avoids pulling in a task scheduler for what
is ultimately a few dozen sessions at most.
"""

from __future__ import annotations

import time
import uuid

SESSION_TTL_SECONDS = 30 * 60  # 30 minutes of inactivity
MAX_HISTORY_MESSAGES = 40      # 20 user/assistant pairs


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, dict] = {}

    def _evict_stale(self) -> None:
        now = time.time()
        stale = [
            sid for sid, s in self._sessions.items()
            if now - s["last_active"] > SESSION_TTL_SECONDS
        ]
        for sid in stale:
            del self._sessions[sid]

    def get_or_create(self, session_id: str | None) -> dict:
        """
        Return the session dict for session_id, creating a new one if it
        doesn't exist or wasn't provided.

        persona / voice_gender / display_name are intentionally NOT
        seeded here from request data. api/chat.py sets them fresh on
        every call instead, rather than trusting a value stored at
        session creation. That's what makes the settings toggle feel
        live: change persona mid-conversation and the very next turn
        uses it.
        """
        self._evict_stale()

        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session["last_active"] = time.time()
            return session

        new_id = session_id or str(uuid.uuid4())
        session = {
            "session_id": new_id,
            "history": [],  # list of {"role": "user"|"assistant", "content": str}
            "persona": "assistant",
            "voice_gender": "female",
            "display_name": None,
            "conversation_count": 0,
            "created_at": time.time(),
            "last_active": time.time(),
        }
        self._sessions[new_id] = session
        return session

    def build_messages_for_llm(
        self,
        system_prompt: str,
        history: list[dict],
        user_message: str,
    ) -> list[dict]:
        """Assemble the full message list for call_llm(), system prompt first."""
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def append_turn(self, session_id: str, user_message: str, reply_text: str) -> tuple[str, str]:
        """
        Record a completed turn and return (user_message_id, message_id),
        generated locally since there's no database row to key off. IDs
        are still returned so the frontend's optimistic-update pattern
        (attaching an id to a bubble once the server responds) works the
        same way it does against production, even though nothing here
        is persisted beyond this process.
        """
        session = self._sessions[session_id]
        session["history"].append({"role": "user", "content": user_message})
        session["history"].append({"role": "assistant", "content": reply_text})
        session["conversation_count"] += 1
        session["last_active"] = time.time()

        if len(session["history"]) > MAX_HISTORY_MESSAGES:
            session["history"] = session["history"][-MAX_HISTORY_MESSAGES:]

        return str(uuid.uuid4()), str(uuid.uuid4())
