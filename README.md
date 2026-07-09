# AI Companion Chatbot Demo

A standalone, voice-first conversational AI demo: a layered system-prompt
architecture, a deterministic safety guardrail with a model-level backstop,
a complete voice pipeline (speech recognition, LLM with live search
grounding, speech synthesis), MCP tool calling, and an LLM-as-Judge
evaluation framework validated with Cohen's kappa against a hand-labeled
gold set.

**Live demo:** https://ying2sun.github.io/ai-companion-chatbot-demo/
*(the first message may take up to a minute, since the backend runs on a
free hosting tier that sleeps after inactivity and has to wake back up)*

## About

[PéiNín Foundation](https://peininfoundation.org/) is a 501(c)(3)
nonprofit building a voice-first AI companion app for Mandarin- and
Cantonese-speaking seniors. I am the AI engineer building that system,
which runs in production on AWS ECS with containerized deployment and
automated CI/CD. This repository is an independent, from-scratch demo of
the same class of system, built so the work can be shared publicly. No
code, prompts, data, or content from the production codebase or any other
product appears here. The real product is at pre-launch stage. An early
web test build is at
[peinin-senior-care.vercel.app](https://peinin-senior-care.vercel.app/).

## Architecture at a glance

```
voice or text turn
        │
        ▼
guardrail pattern check ──match──▶ fixed safety reply
        │                          (single-digit ms, no model call)
        ▼ no match
Groq Whisper STT (audio turns only)
        │
        ▼
3-layer prompt assembly
(identity + safety │ persona tone │ session context)
        │
        ▼
Gemini, search grounding ◀──▶ MCP unit-conversion tool
        │
        ▼
deterministic post-processing
(suggestion chips, unit conversion detection)
        │
        ▼
MiniMax TTS ──▶ reply text + audio + guardrail metadata
```

## Skills demonstrated

- **Validated LLM-as-Judge evaluation.** A hybrid framework: deterministic
  checks (`eval/checks.py`, runnable with no API key) for mechanically
  verifiable items, plus a cross-family LLM judge (the judge is Claude,
  the system under test is Gemini) validated against a 27-turn
  hand-labeled gold set. On the rubric item with real statistical
  support, tone-persona match, the judge reached kappa = 0.899 (n = 27,
  one disagreement). The gold set's own labels were independently
  cross-checked by GPT-4o and DeepSeek V3, which overturned one of my
  original labels. The disagreement is documented in the gold set rather
  than silently resolved. Full numbers and caveats under *Evaluation*.
- **Deterministic safety layer with a measurable fast path.** Crisis and
  medication language is caught by pattern matching before any model
  call. A triggered turn returns a fixed, careful response in single-digit
  milliseconds, against several hundred for a normal reply, observable
  live in the debug panel. The model-level instruction is a backstop, and
  the known limitation of this design is stated plainly below.
- **Layered prompt architecture with isolation guarantees.** System
  instructions are split into three independently swappable layers (fixed
  identity and safety rules, per-conversation persona tone, session
  context), assembled fresh each turn. Changing the tone cannot break a
  safety rule because the two never live in the same string.
- **Tool calling via MCP, confirmed live in production.** Unit
  conversion is exact arithmetic, so Gemini calls a local MCP server
  (FastMCP) mid-generation whenever it judges a direct conversion
  question needs one, for example "convert 90 kilograms to pounds." This
  is not just a local demo of the protocol, it fires in the deployed app,
  and it is the same call path that caused a roughly 15-second latency
  spike I traced back to two real round trips to Gemini plus a real
  subprocess call to the MCP server. Its job is narrow, making sure a
  number inside Gemini's own generated reply is exact rather than
  guessed. A separate, unconditional path, `detect_unit_chips`, scans
  every finished reply for any measurement no matter how it got there,
  whether from the tool or from Gemini's own training knowledge, and
  turns it into a visible chip under the message. The two paths overlap
  in exactly one case: a measurement Gemini states from its own
  knowledge rather than a computed conversion never touches the MCP path
  at all, and text detection is the only thing that surfaces it.
- **Production-minded engineering at demo scale.** Two independent
  rate-limiting layers (a per-visitor limit and an unrelated global daily
  cap, since every turn triggers three paid API calls), defensive
  handling of malformed credentials, and a deliberately stateless,
  no-login architecture sized for what this is.

## Evaluation

Testing an LLM-backed feature is not like testing ordinary code: the same
input can produce many reasonable replies, and some of what matters most
(did the reply acknowledge a feeling before offering advice, did it avoid
correcting the person) cannot be verified by a regex. The framework here
splits the problem accordingly: rubric items with a mechanically checkable
answer get plain deterministic functions in `eval/checks.py`, fast and
free to run. Items that genuinely require judgment go to an LLM judge
scored against a written rubric.

The judge is deliberately a different model family from the system under
test (Claude judging Gemini), so the grading cannot drift toward
self-preference bias, the documented tendency of a model to favor outputs
that sound like its own.

### Judge validation results

A judge is only useful if it demonstrably agrees with a careful human
reviewer, so it is validated against a 27-turn hand-labeled gold set
using Cohen's kappa, which corrects raw agreement for what chance alone
would produce. The conventional bar for trusting a judge is kappa above
roughly 0.6.

| Rubric item | n | Raw agreement | Kappa | |
|---|---|---|---|---|
| tone_matches_persona | 27 | 96.3% | 0.899 | pass |
| emotional_first | 6 | 100% | 1.000 | pass |
| ai_honesty | 5 | 100% | 1.000 | pass |
| non_judgment | 5 | 100% | 1.000 | pass |

The tone item is the meaningful validation: 27 turns, one disagreement,
kappa well above the bar. The other three items apply to fewer turns, and
perfect agreement at n = 5 or 6 is consistent with a working judge but
too small a sample to certify one. Growing those slices of the gold set
is the obvious next step, and I would rather state that than present
three 1.000s as strong evidence.

> **Note:** these figures are from the initial validation run
> (July 2026). The judge will be re-validated after the gold set is
> expanded alongside the production evaluation work for PéiNín, and this
> table will be updated then.

### Severity tiers

Not every failure matters equally, so every check carries a severity,
mirroring the three-tier idea in the runtime guardrail itself:
safety-critical failures block release outright, major failures are held
to a pass-rate threshold, minor failures are tracked over time without
blocking anything. A single averaged score would bury a rare but serious
failure inside a sea of harmless ones.

### The gold set itself gets cross-checked, not just the judge

Kappa validation answers whether the judge can be trusted against the
gold set. It says nothing about whether the gold set's own labels are
right, and a set labeled by one person has had only one point of view
applied to it. So the labels were independently re-scored by two models
from different families (GPT-4o and DeepSeek V3, via OpenRouter) against
the exact same rubric. Agreement between the human label and both models
counts as independent confirmation. Disagreement flags a specific, named
turn for a second look.

This surfaced a real case: one reply took a clear side on a sensitive
family matter (already failing on that basis) but was originally labeled
as still matching its persona's direct tone, on the reasoning that
delivery style and content are separate dimensions. GPT-4o and a
tie-breaking third model (Grok) both independently argued the bluntness
itself broke the persona's tone. The label was updated to that consensus,
and the disagreement is left documented in the gold set's notes rather
than quietly resolved.

### Scope of what is shared

This framework demonstrates the same structure as the evaluation system I
designed for PéiNín's production chatbot: hybrid deterministic-plus-judge
checks, the same severity model, the same kappa-based validation. Every
test case, conversation, and finding here was written fresh for this
demo. Nothing is drawn from real conversations or real production
findings, which are exactly the kind of material that should not be
public. One further boundary worth naming: replies grounded by live
search are not independently fact-verified in this demo. That
verification layer is deliberately out of scope here.

## How it works

### Layered instructions instead of one long prompt

The model's instructions are assembled fresh every turn from three
pieces. A fixed layer defines what the AI is, its honesty requirements,
and reply formatting. A persona layer sets the tone for the conversation
and can be swapped without touching anything else. A context layer adds
whatever the system currently knows about this specific session, and is
omitted entirely when empty. The payoff is isolation: a tone adjustment
cannot accidentally weaken a safety rule.

Concretely, an early version of the how-to instruction told the model to
give a brief overview of the general approach. That was reworded to give
the general steps, or a couple of main directions. Tested before and
after on the same kind of question, an appliance-troubleshooting request,
the difference was not subtle. The first wording produced a vague,
high-level response. The second named the actual sequence to check, the
circuit breaker, then a hard reset, then a possible blown fuse. This was
a direct before-and-after comparison, not a controlled study, but it is
exactly the kind of thing a layered prompt structure lets you isolate and
test one piece at a time.

### Catching a crisis before the model ever sees it

Some messages, medication questions or language suggesting someone may be
in crisis, need a careful response every time, and a well-instructed
model follows instructions reliably rather than perfectly. So the system
checks first: a curated set of words and phrases is matched against the
message before any model call, and a hit returns a fixed, careful
response immediately. The model's own instructions remain as a second
line of defense for anything the patterns miss.

The known limitation is stated rather than hidden: keyword matching
trades recall for determinism. It cannot catch paraphrases it has never
seen, and the pattern set is the component that would need the most
ongoing curation in a real deployment. That trade is deliberate here,
because the failure mode of a fixed safe response is mild and the failure
mode of a missed crisis signal is not.

### Giving the model a calculator instead of letting it guess

Language models are unreliable at exact arithmetic, and a unit conversion
has one correct answer with no room for a plausible wrong one. The
arithmetic therefore lives in plain code. It runs as deterministic
detection on reply text, and the same logic is exposed as a tool on a
local MCP server (FastMCP) that a model client can discover and call
while generating a reply. The wider habit this demonstrates: know which
parts of a task the model should own and which parts belong to something
mechanically reliable.

### Two rate limits for two failure modes

A per-visitor limit stops the obvious abuse case. It does nothing about
many separate visitors each sending a handful of messages, which adds up
to real cost with no single sender looking abusive, and every turn here
triggers three paid API calls. A second, independent global daily cap
covers that case, so the project has a hard ceiling regardless of how
usage is spread.

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
ai-companion-chatbot-demo/
├── backend/
│   ├── api/
│   │   └── chat.py           # the /chat endpoint
│   ├── core/
│   │   └── limiter.py        # rate limiting
│   ├── eval/
│   │   ├── checks.py         # deterministic evaluation checks
│   │   ├── gold_set.json     # 27 hand-labeled turns
│   │   ├── judge.py          # Claude-based LLM judge
│   │   ├── validate_judge.py # kappa validation against the gold set
│   │   ├── label_with_models.py # cross-checks gold set labels via OpenRouter
│   │   └── compare_labels.py # reports where human/GPT-4o/DeepSeek disagree
│   ├── llm/
│   │   ├── client.py         # Gemini integration
│   │   ├── guardrails.py     # safety layer
│   │   └── prompts.py        # three-layer prompt builder
│   ├── mcp_tools/
│   │   └── units_server.py   # MCP server exposing unit conversion as a tool
│   ├── sessions/
│   │   └── store.py          # in-memory session store
│   ├── stt/
│   │   └── service.py        # Groq Whisper integration
│   ├── tts/
│   │   ├── google_service.py # MiniMax integration (file name is legacy from an earlier provider)
│   │   └── service.py
│   ├── suggestions/
│   │   ├── chips.py          # phone number / URL detection
│   │   └── units.py          # unit conversion detection and math
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

`GET /health` should return `{"status": "ok"}`. FastAPI serves interactive
API docs at `/docs` once the server is running.

Running the evaluation: `python eval/checks.py` from `backend/` needs no
API key. `python eval/validate_judge.py --self-test` verifies the kappa
math itself, also with no key. A live judge run needs `ANTHROPIC_API_KEY`.
The gold-set cross-check needs `OPENROUTER_API_KEY` (run
`label_with_models.py`, then `compare_labels.py`).

The frontend is a single static file, `docs/index.html`. To run it against
a local backend, open it directly in a browser and point the `API_BASE`
constant near the top of its script at `http://localhost:8000`. The
deployed copy is already pointed at the live backend.

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
| Evaluation | Complete, judge validated against the gold set (see results table above), re-validation scheduled after the gold set expands |

## Scope and boundaries

This is an independent, from-scratch implementation. No code, prompts,
data, or content from any other codebase or product was used in building
it.

**No persistence, by design.** There is no database. A conversation lives
in memory only while the tab is open and the session is active. Tab
close, a 30-minute idle timeout, or a server restart clears it. Server
logs record only metadata (session IDs, timing, character counts), never
message or reply text. For a no-login, try-it-once demo this is exactly
enough. For something people depend on daily, it would be the first
thing to change.

## License

Not currently licensed for reuse. If you would like to use any of this,
open an issue.
