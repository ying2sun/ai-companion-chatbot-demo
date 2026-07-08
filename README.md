# AI Companion Chatbot Demo

A standalone, voice-first conversational AI backend and frontend: a layered
system-prompt architecture, a two-tier safety guardrail, a full speech in,
speech out pipeline, and an evaluation framework to score it, built to
demonstrate AI engineering, not to ship a product.

**Live demo:** https://ying2sun.github.io/ai-companion-chatbot-demo/

## About

This is a demo for the AI chatbot part of [PéiNín
Foundation](https://peininfoundation.org/). I'm the AI engineer building that
system, and put this independent, from-scratch demo together to share
similar work publicly. The real product is at pre-launch stage; you can
check out a web test version only, not the production experience, at
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

## Evaluation

Testing an LLM-backed feature isn't like testing ordinary code. There's no
single expected output to assert equality against, the same input can
produce many reasonable replies, and some of the things that actually
matter (did the reply acknowledge the person's feeling before offering
advice, did it avoid correcting them) aren't the kind of thing a regex can
verify. This project's evaluation approach is built around that difference.

### Two kinds of checks, for two kinds of questions

Some rubric items have a mechanically checkable answer: does the reply
contain markdown formatting, does it ask more than one question, did the
safety layer fire on a message written specifically to trigger it. Those
get plain functions, `eval/checks.py` in this repo, fast, deterministic,
and free to run as often as needed.

Other items are inherently a judgment call: does the first sentence read as
acknowledging a feeling rather than jumping to advice, does the tone feel
warm rather than scripted. No pattern list captures that reliably. Those
need a second model to read the conversation and score it against a
rubric, an LLM judge. Running both kinds together means the parts of
quality that are mechanically checkable get checked cheaply and reliably,
and the parts that genuinely require judgment get evaluated by something
capable of it.

### Why the judge has to be a different model from the one being graded

If the same model family were both the system under test and its own
judge, the judge could end up subtly favoring outputs that sound like its
own style, a self-preference bias, rather than genuinely grading against
the rubric. This project's chat model is Gemini; the planned judge is a
different model family entirely, specifically so the grading is
independent of the thing being graded, the same principle behind having a
second person review work rather than letting someone grade their own
exam.

### Validating the judge itself, before trusting it

A judge is only useful if there's actual evidence it agrees with what a
careful human reviewer would say, not just evidence that it agrees with
itself. The standard way to check this: hand-label a small set of example
conversations first (a gold set), have the judge score the same
conversations, then compare the two sets of labels.

Simple percent agreement is misleading here. If 95% of ordinary
conversations are trivially fine on some item, a judge that says "pass"
on every single turn also achieves 95% agreement, without having actually
evaluated anything. Cohen's kappa corrects for this: it measures how much
agreement exceeds what pure chance would already produce given each
rater's overall tendencies. A kappa near zero means a judge added no real
signal even when raw agreement looked strong. A kappa above roughly 0.6 is
generally treated as strong enough evidence to trust that judge's scores
going forward.

### Severity, not one blended score

Not every failure matters equally. A reply that runs slightly longer than
ideal is a different category of problem from one that misses language
associated with someone in crisis. This project tags every check with a
severity, mirroring the same three-tier idea used in the runtime guardrail
itself: safety-critical failures block release outright, major failures
are held to a pass-rate threshold, minor failures are tracked over time
without blocking anything. A single averaged score would bury a rare but
serious failure inside a sea of harmless ones.

### What this demonstrates, and what it deliberately excludes

This evaluation framework is a demo of the real evaluation system I
designed for PéiNín's chatbot: the same hybrid deterministic-plus-judge
structure, the same severity model, the same kappa-based judge validation.
What's here is deliberately not a copy of that system. Every test case,
example conversation, and finding in this repository was written fresh for
this demo, none of it is drawn from real conversations, real test phrases,
or real findings from the production system. Some of what a real
evaluation run surfaces is exactly the kind of information that shouldn't
be public regardless of where it comes from, and none of it is needed to
demonstrate the methodology itself.

### Status

| Piece | Status |
|---|---|
| Deterministic checks (`eval/checks.py`) | Complete, runnable now with no API key |
| Gold-set-validated LLM judge | Planned |
| Kappa validator | Planned |

## Architecture

| Layer | Choice | Notes |
|---|---|---|
| Backend | FastAPI | Stateless, in-memory session store, no database |
| LLM | Gemini (`google-genai` SDK) | Search grounding enabled for current-events questions |
| Speech-to-text | Groq (Whisper large-v3) | Managed API, no cold start |
| Text-to-speech | MiniMax Speech-02-HD | Multiple voice and tone combinations |
| Rate limiting | slowapi | Per-IP limit plus an independent global daily cap |
| Frontend | Plain HTML/CSS/JS | No build step, no framework |
| Hosting | Render (backend) + GitHub Pages (frontend) | Free tiers |

## Project structure

```
peinin-ai-demo/
├── backend/
│   ├── api/
│   │   └── chat.py          # the /chat endpoint
│   ├── core/
│   │   └── limiter.py       # rate limiting
│   ├── eval/
│   │   └── checks.py        # deterministic evaluation checks
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
├── docs/                     # frontend, served directly by GitHub Pages
│   └── index.html
├── postman/                  # test collection, see Testing below
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

The frontend is a single static file, `docs/index.html`. To run it against
a local backend, open it directly in a browser and make sure the
`API_BASE` constant near the top of its script points at
`http://localhost:8000`. Against the deployed backend, it's already
pointed there, that's what's live at the URL at the top of this README.

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

For the evaluation checks specifically, `python eval/checks.py` runs a full
self-test with built-in fixture cases and needs no API key at all.

## Status

| Component | Status |
|---|---|
| Backend | Complete, deployed |
| Frontend | Complete, deployed |
| Evaluation | In progress, see Evaluation above |

## Scope and boundaries

This is an independent, from-scratch implementation. No code, prompts,
data, or content from any other codebase or product was used in building
it.

## License

Not currently licensed for reuse. If you'd like to use any of this, open
an issue.
