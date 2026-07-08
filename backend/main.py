"""
backend/main.py
-----------------
Demo app entry point.

What's dropped from production, and why: production registers thirteen
routers (activity, users, news, transcribe, profile, audio, conversation,
history, push, speech, engagement, daily_starter, plus chat). Every one
of them except chat depends on Supabase, S3, or a product feature this
build doesn't have (family dashboard, medication reminders, news
pipeline, engagement scoring persistence). There's nothing for them to
do here, so they're not imported at all rather than imported and left
broken.

No Firebase, no local static-file mount for audio: production serves a
/local-audio path for dev-mode file playback before S3 is configured.
This build never writes audio to disk, TTS bytes go straight into the
JSON response, so there's no local storage directory to serve.
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
