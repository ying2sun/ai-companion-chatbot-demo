"""
backend/main.py
-----------------
Demo app entry point.

A fuller production app would register additional routers for
persistent, database-backed features (profile, history, family
features, and similar) that have no equivalent in this stateless demo,
so they're not imported at all rather than imported and left broken.

No local static-file mount for audio either: this build never writes
audio to disk, TTS bytes go straight into the JSON response, so
there's no local storage directory to serve.
"""

import logging

from dotenv import load_dotenv

load_dotenv()  # must run before anything else reads env vars

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(name)s: %(message)s",
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.chat import router as chat_router
from core.limiter import limiter

app = FastAPI(title="AI Companion Demo Backend", version="0.1.0")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ── CORS ─────────────────────────────────────────────────────────────────────
# Wide open is fine here: no cookies, no auth, no origin-bound session.
# Tighten allow_origins to your deployed frontend's actual origin once
# you know it, if you'd rather not leave it open.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}
