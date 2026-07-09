# AI Companion Chatbot Demo

A standalone, voice-first conversational AI backend and frontend: a layered
system-prompt architecture, a two-tier safety guardrail, a full speech in,
speech out pipeline, and an evaluation framework to score it, built to
demonstrate AI engineering, not to ship a product.

**Live demo:** https://ying2sun.github.io/ai-companion-chatbot-demo/
*(the first message may take up to a minute, the server is on a free
hosting tier that goes to sleep after inactivity and has to wake back up)*

## About

PéiNín Foundation is a 501(c)(3) nonprofit organization building a
voice-first AI companion app for Mandarin- and Cantonese-speaking seniors.
This is a demo for the AI chatbot part of [PéiNín
Foundation](https://peininfoundation.org/). I'm the AI engineer building
that system, and put this independent, from-scratch demo together to share
similar work publicly. The real product is at pre-launch stage. You can
check out an early web test build at
[peinin-senior-care.vercel.app](https://peinin-senior-care.vercel.app/).

## Skills demonstrated

- **Prompt engineering.** A layered system prompt (identity, tone,
  session memory) instead of one large block, with enough attention to
  exact wording that a single word choice was found, through testing, to
  measurably change model behavior. See *How it works*.
- **AI safety engineering.** A two-tier guardrail, deterministic pattern
  matching backed by a model-level instruction, engineered to skip the
  LLM call entirely when it fires. Observable live in the debug panel:
  a guardrail-triggered turn runs in single-digit milliseconds against
  several hundred for a normal reply.
- **Matching computation to the task.** People often want a converted
  figure when a reply states a measurement (e.g. pounds vs. kilograms,
  Fahrenheit vs. Celsius), a real and common need in conversation.
  Rather than asking the model to do that arithmetic, unreliable in a
  way a one-line formula is not, unit conversions are detected in the
  text and computed with plain, deterministic code, so the number
  returned is always exactly right.
- **Tool-calling architecture via MCP.** That same conversion logic is
  also exposed as a real callable tool using the Model Context Protocol,
  so the model can reach for it directly while generating a reply
  instead of only being caught after the fact. Verified against the
  actual protocol, a real MCP server discovered and called through a
  live client connection.
- **Evaluation methodology.** A framework built around the same
  principles as rigorous production ML evaluation: deterministic checks
  where correctness is checkable by code, an LLM judge for what genuinely
  requires judgment, and validating that judge against a hand-labeled
  gold set with Cohen's kappa rather than trusting it blindly. See
  *Evaluation*.
- **Multi-model orchestration.** Three separate specialized APIs,
  speech-to-text, a language model with live search grounding, and
  text-to-speech, coordinated into one coherent, stateful conversation.
- **Cloud infrastructure at production scale.** This demo runs on free
  hosting suited to its purpose. The production system it's modeled on
  runs on AWS ECS, a real, scaled deployment I also built and operate.
- **Production-minded engineering, at the right scale for what this is.**
  Two independent rate-limiting layers, defensive handling of malformed
  credentials, a stateless architecture sized deliberately for a
  no-login demo rather than over-built.
- **Real testing discipline.** DOM-level functional
  tests for the frontend, unit and integration tests for the guardrail,
  chip detection, and the chat endpoint itself, run and passing before
  anything shipped.

## How it works

### Layered instructions instead of one long prompt

The instructions given to the AI model are split into three pieces,
assembled fresh every time a reply is generated. One piece never
changes: what the AI is, how honest it needs to be, how a reply should
be formatted. A second piece sets the tone for this particular
conversation, warmer and closer, or more businesslike, and swapping it
never touches anything else. A third piece adds whatever the system
currently knows about this specific conversation, left out entirely
when there's nothing yet to add.

The payoff shows up when something needs to change later. Adjusting the
tone can't accidentally break a safety rule, because the two live in
genuinely separate places.

### Catching a crisis before the model ever sees it

Some messages need a careful response no matter what, a medication
question, or language suggesting someone might be in crisis. Trusting
the AI model to always catch these on its own isn't good enough. Even a
well-instructed model follows its instructions reliably, not perfectly,
and this is exactly the place where "reliably" isn't the bar.

So the system checks first, before the model is ever called. A specific
set of words and phrases known to signal one of these situations gets
checked against the message directly, and a match sends a fixed,
careful response immediately, no model judgment involved at all. The
model's own instructions are still there as a second line of defense
for anything the check misses, but the cases that matter most never
depend on it getting that right in the moment.

### What actually happens when you talk instead of type

Three separate steps happen behind a spoken exchange. The recording
gets turned into text first. That text goes to the AI model, which
writes a reply. The reply gets turned back into audio. Three different
jobs, three different purpose-built services handling them, rather than
one system trying to do all three itself.

None of those three services are built in-house here. Running that kind
of technology yourself is real, ongoing infrastructure work, and none
of it would make the actual conversation better, only more expensive to
keep running. Using hosted versions instead means each one keeps
improving on its own, without this project needing to touch it.

### Giving the model a calculator instead of letting it guess

Language models are good at language and unreliable at exact
arithmetic, the same way someone doing long division in their head is
more error-prone than someone using a calculator. Converting units is
exactly that kind of task: the correct answer is a specific number, and
there's no room for a plausible-sounding wrong one.

Rather than trust the model to compute a conversion itself, this
project gives it an actual tool it can call while writing a reply, a
small local server, built with the Model Context Protocol (MCP), that
does the arithmetic in plain code and hands back the exact result. Ask
directly how many pounds is 90 kilograms, and the model can reach for
the tool instead of guessing. It's a small example of a bigger habit:
know which parts of a task the model should own, and which parts belong
to something actually reliable at them.

### Nothing is remembered on purpose

There's no database behind this project. A conversation exists in
memory for as long as the browser tab stays open and the session stays
active. Closing the tab, 30 minutes of inactivity, or the server
restarting all clear it. There's no login here, and nothing about a
demo like this calls for remembering someone after they're done with
it.

The honest trade-off: this would be a real problem for something people
depend on daily. For something meant to be tried once, it's exactly
enough.

### Two different ways this could be overwhelmed, two different limits

Limiting how often one visitor can send a message stops the obvious
case. It does nothing about a quieter one: a lot of separate people,
each sending only a handful of messages, adding up to real cost without
any single one of them looking like abuse.

Two independent checks run here for that reason. One watches
each visitor. A second, unrelated to the first, watches total usage
across everyone combined, since every message here triggers three
separate paid API calls, and a demo project deserves a hard ceiling on
that regardless of how the usage is spread out.

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
the rubric. This project's chat model is Gemini. The planned judge is a
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

### Why the gold set itself gets cross-checked, not just the judge

Everything above validates whether the judge can be trusted against the
gold set. It says nothing about whether the gold set's own labels are
right in the first place. A gold set written by one person, even
carefully, with reasoning noted for the tricky cases, has only had one
point of view applied to it.

So the gold set's labels get checked too, independently, using two
different models (GPT-4o and DeepSeek V3, reached through OpenRouter)
scored against the exact same rubric the judge uses. Where a human
label and both models agree, that's real, independent confirmation.
Where they don't, that's a specific, named turn worth a second look,
not a vague sense that something might be off. This is a different
question from judge validation, one asks whether the judge can be
trusted, this one asks whether the thing the judge is being validated
against can be trusted, and skipping it would leave that second
question completely unexamined.

This isn't hypothetical. Running it against this gold set surfaced a
real disagreement: one turn where a reply took a clear side on a
sensitive family matter (already failing on that basis) but was
originally labeled as still matching its assigned persona's direct
tone, on the reasoning that delivery style and content are separate
dimensions. GPT-4o and a third model (Grok, used as a tie-breaker)
both independently pushed back, arguing the bluntness itself broke the
persona's tone, not just the underlying judgment. The label was
updated to match that consensus, and the disagreement is left
documented in the gold set's notes rather than quietly resolved, a
defensible eval system should be able to show its work, not just its
conclusions.

### Status

| Piece | Status |
|---|---|
| Deterministic checks (`eval/checks.py`) | Complete, runnable now with no API key |
| Gold set (`eval/gold_set.json`) | Complete, 27 original turns |
| LLM judge (`eval/judge.py`) | Complete, needs `ANTHROPIC_API_KEY` to run against real Claude output |
| Kappa validator (`eval/validate_judge.py`) | Complete, its own math is verified independently of the judge, run `python validate_judge.py --self-test` |
| Gold-set cross-check (`eval/label_with_models.py`, `eval/compare_labels.py`) | Complete, needs `OPENROUTER_API_KEY` to run against real GPT-4o and DeepSeek V3 output |

Run it yourself: `python eval/checks.py` from `backend/`, no API key needed.
The judge and validator need a real `ANTHROPIC_API_KEY` to produce an
actual kappa number, `python eval/validate_judge.py --self-test`
verifies the kappa math itself with no key required at all. The
gold-set cross-check needs a real `OPENROUTER_API_KEY`, run
`label_with_models.py` first, then `compare_labels.py` to see the
result.

## Architecture

| Layer | Choice | Notes |
|---|---|---|
| Backend | FastAPI | Stateless, in-memory session store, no database |
| LLM | Gemini (`google-genai` SDK) | Search grounding enabled for current-events questions |
| Tool calling | MCP (FastMCP) | Local server exposing unit conversion as a callable tool |
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
│   │   ├── checks.py         # deterministic evaluation checks
│   │   ├── gold_set.json     # 27 hand-labeled turns
│   │   ├── judge.py          # Claude-based LLM judge
│   │   ├── validate_judge.py # kappa validation against the gold set
│   │   ├── label_with_models.py # cross-checks gold set labels via OpenRouter
│   │   └── compare_labels.py # reports where human/GPT-4o/DeepSeek disagree
│   ├── llm/
│   │   ├── client.py        # Gemini integration
│   │   ├── guardrails.py    # safety layer
│   │   └── prompts.py       # three-layer prompt builder
│   ├── mcp_tools/
│   │   └── units_server.py  # MCP server exposing unit conversion as a tool
│   ├── sessions/
│   │   └── store.py         # in-memory session store
│   ├── stt/
│   │   └── service.py       # Groq Whisper integration
│   ├── tts/
│   │   ├── google_service.py # MiniMax integration
│   │   └── service.py
│   ├── suggestions/
│   │   ├── chips.py         # phone number / URL detection
│   │   └── units.py         # unit conversion detection and math
│   ├── main.py
│   ├── requirements.txt
│   └── .env.example
├── docs/                     # frontend, served directly by GitHub Pages
│   └── index.html
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

## Status

| Component | Status |
|---|---|
| Backend | Complete, deployed |
| Frontend | Complete, deployed |
| Evaluation | Complete, live kappa run pending a real API key, see Evaluation above |

## Scope and boundaries

This is an independent, from-scratch implementation. No code, prompts,
data, or content from any other codebase or product was used in building
it.

**No persistence.** Conversation history isn't stored anywhere, no
database, no file. Everything lives in memory for the length of a
session and is lost on tab close, a 30-minute idle timeout, or a server
restart, which happens automatically on this hosting tier after a period
of inactivity. Server logs record only metadata (session IDs, timing,
character counts), never the actual message or reply text, so even the
server's own logs don't reveal what anyone said.

## License

Not currently licensed for reuse. If you'd like to use any of this, open
an issue.
