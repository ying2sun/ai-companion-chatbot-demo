# AI Companion Chatbot Demo

A standalone, voice-first conversational AI backend and frontend: a layered
system-prompt architecture, a two-tier safety guardrail, and a full speech
in, speech out pipeline, built to demonstrate AI engineering, not to ship a
product.

## About

This is a demo for the AI chatbot part of [PéiNín
Foundation](https://peininfoundation.org/). The real product is at
pre-launch stage; you can check out a web test version only, not the
production experience, at
[peinin-senior-care.vercel.app](https://peinin-senior-care.vercel.app/).

## How it works

### Why the system prompt is layered instead of written as one block

As a prompt accumulates rules (identity, tone, formatting constraints,
safety behavior, context about the current session) a single unstructured
block of instructions gets harder to edit safely over time. Every new rule
risks colliding with an existing one, and it becomes difficult to tell, just
by reading the prompt, which sentence is responsible for which behavior.

This project splits the prompt into three layers that are assembled in a
fixed order rather than written as one document:

- **A base layer**, applied to every conversation regardless of anything
  else: identity, core behavioral rules, formatting constraints.
- **A tone layer**, selected per session, that adjusts the relationship
  register (for example, a closer, warmer tone versus a more neutral,
  task-focused one) without touching anything in the base layer.
- **A memory layer**, built fresh from whatever context exists about the
  current session. When there's no context yet, this layer renders as an
  empty string rather than a templated placeholder, so a brand-new session
  gets a clean prompt with no awkward gap.

Keeping the layers separate means each one can be edited, tested, or
swapped independently. Changing the tone layer can't accidentally break a
safety rule that lives in the base layer, because they're different
functions composed together, not different paragraphs of the same text.

### Why there are two guardrail layers, not one

A common approach to keeping an LLM on safe ground is a single instruction
in the system prompt telling the model what to avoid or redirect. That
works, but it's probabilistic: the model follows instructions with high but
not perfect reliability, every turn pays the token and latency cost of that
instruction whether or not it's ever needed, and when something does go
wrong there's no way to point to the specific rule that did or didn't fire.

This project runs a deterministic check first, before the language model is
ever called. A small set of pattern-matching rules scans the person's own
message for language associated with a small number of high-stakes
categories. If a pattern matches, the language model call is skipped
entirely and a fixed, pre-written response is returned instead. This layer
is fast (sub-millisecond), free (no API call made), and fully auditable,
every rule is a plain, readable line of code, not a probability.

Its weakness is recall: a fixed pattern list can't anticipate every way a
person might phrase something. So the system prompt itself carries a
second, complementary instruction as a backstop for phrasing the pattern
layer misses. Neither layer is sufficient alone. The deterministic layer
catches the common, high-confidence cases cheaply and predictably; the
model-level instruction catches the long tail the patterns can't
anticipate.

### The voice pipeline

A spoken turn moves through three separate specialized models in sequence:
speech-to-text turns the incoming audio into a transcript, a language model
generates a reply from that transcript plus conversation history, and
text-to-speech turns the reply back into audio. Each stage is called
through a managed API rather than a self-hosted model.

That's a deliberate trade-off. Self-hosting any of these models means
owning GPU provisioning, cold-start latency after idle periods, and scaling
as usage grows, real infrastructure work that adds nothing to the actual
conversation quality. Managed APIs move all of that off this codebase
entirely, at the cost of a small amount of added per-call latency and no
control over the exact model version in use. For a project at this stage,
that trade favors managed APIs by a wide margin: it keeps the codebase
focused on orchestration, and lets underlying model quality improve
automatically as each provider updates their service, rather than the
project being pinned to a specific self-hosted model version that only gets
better when someone manually updates it.

### Stateless by design

This backend keeps no database. A conversation exists entirely in memory,
keyed by a session ID generated client-side on page load, and is dropped on
a time-based expiry if the session goes idle. That's a deliberately
right-sized amount of infrastructure, not a missing feature: there's no
account system, no login, and no reason for a conversation to outlive the
browser tab it happened in. State doesn't survive a server restart or a
closed tab, which would be unacceptable for a product people rely on day to
day, but is exactly the right amount of durability for something meant to
be tried, not depended on.

### Two independent rate limits

A single limit on requests per IP address stops one obvious source from
overwhelming the service, but it's blind to a different failure mode: a
slow trickle of requests arriving from many different sources, which never
looks abusive from any single IP's perspective even as it adds up in
aggregate. This project runs two independent checks for that reason: a
per-IP limiter handles the first case, and a separate global counter,
tracked independently of where requests originate, acts as a hard ceiling
on total usage regardless of pattern. Either check can reject a request on
its own; neither depends on the other being present.

## Architecture

| Layer | Choice | Notes |
|---|---|---|
| Backend | FastAPI | Stateless, in-memory session store, no database |
| LLM | Gemini (`google-genai` SDK) | Search grounding enabled for current-events questions |
| Speech-to-text | Groq (Whisper large-v3) | Managed API, no cold start |
| Text-to-speech | MiniMax Speech-02-HD | Multiple voice and tone combinations |
| Rate limiting | slowapi | Per-IP limit plus an independent global daily cap |
| Frontend | Plain HTML/CSS/JS | No build step, no framework |

## Project structure

```
peinin-ai-demo/
├── backend/
│   ├── api/
│   │   └── chat.py          # the /chat endpoint
│   ├── core/
│   │   └── limiter.py       # rate limiting
│   ├── llm/
│   │   ├── client.py        # Gemini integration
│   │   ├── guardrails.py    # safety layer
│   │   └── prompts.py       # three-layer prompt builder
│   ├── sessions/
│   │   └── store.py         # in-memory session store
│   ├── stt/
│   │   └── service.py       # Groq Whisper integration
│   ├── tts/
│   │   ├── google_service.py # MiniMax integration
│   │   └── service.py
│   ├── suggestions/
│   │   └── chips.py         # phone number / URL detection
│   ├── main.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/                 # in progress
├── postman/                  # test collection, see below
├── .gitignore
└── README.md
```

## Getting started

Requires Python 3.11+ and API keys for Gemini, Groq, and MiniMax.

```bash
cd backend
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # then fill in your own keys
uvicorn main:app --reload --port 8000
```

`GET /health` should return `{"status": "ok"}`. FastAPI also serves
interactive API docs at `/docs` once the server is running.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check |
| `/chat` | POST | Text or voice turn, `multipart/form-data` |

`/chat` accepts `message` (text) or `audio` (file upload), plus `persona`,
`voice_gender`, and an optional `session_id` to continue an existing
conversation. It returns `reply_text`, base64-encoded TTS audio, any
detected suggestion chips, and guardrail metadata.

## Testing

A Postman collection covering the happy path, both guardrail categories,
input validation, session continuity, and rate limiting lives in
`postman/`. Import both files, point the environment's `base_url` at your
running server, and run the collection top to bottom.

## Status

| Component | Status |
|---|---|
| Backend | Complete |
| Frontend | In progress |
| Deployment | Not yet deployed |

## Scope and boundaries

This is an independent, from-scratch implementation. No code, prompts,
data, or content from any other codebase or product was used in building
it.

## License

Not currently licensed for reuse. If you'd like to use any of this, open
an issue.
