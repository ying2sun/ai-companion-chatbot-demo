"""
backend/core/limiter.py
-------------------------
Two-layer rate limiting for the demo: per-IP (slowapi, in-memory) and a
global daily cap across every session combined. Both live in one shared
module so main.py and api/chat.py can use the same limiter instance
without importing from each other.

Why two layers, not one:
  Per-IP limiting stops one source from hammering the endpoint. It does
  nothing about a slow trickle of requests arriving from many different
  IPs, which per-IP limiting cannot see as a pattern at all. The daily
  cap is a separate, independent backstop against that: this endpoint
  calls three paid APIs per turn (Gemini, Groq, MiniMax), so a public
  URL, however unlisted, deserves a hard ceiling on total daily spend
  regardless of where requests come from.

Why in-memory for both, not Redis:
  Single-process deployment, no horizontal scaling. An in-memory limiter
  loses its state on restart, which for a demo is a rare, deliberate
  event, not a gap worth adding infrastructure to close.
"""

import datetime
import os

from slowapi import Limiter
from slowapi.util import get_remote_address

# Per-IP limiter. Applied to /chat via a decorator in api/chat.py, not
# globally, since /chat is the only route that costs money to serve.
limiter = Limiter(key_func=get_remote_address)

# Global daily cap. Independent of the per-IP limiter above, see module
# docstring. Override with the DAILY_CHAT_LIMIT env var if you want a
# different ceiling.
_DAILY_LIMIT = int(os.getenv("DAILY_CHAT_LIMIT", "300"))
_daily_state = {"date": None, "count": 0}


def check_daily_cap() -> bool:
    """
    Returns True if today's global /chat budget still has room, False
    if the cap has been reached. Resets automatically when the date
    rolls over (UTC), no scheduled job needed.
    """
    today = datetime.date.today().isoformat()
    if _daily_state["date"] != today:
        _daily_state["date"] = today
        _daily_state["count"] = 0

    if _daily_state["count"] >= _DAILY_LIMIT:
        return False

    _daily_state["count"] += 1
    return True
