"""
backend/eval/label_with_models.py
------------------------------------
Cross-checks the gold set's human labels using two independent models,
GPT-4o and DeepSeek V3, both reached through OpenRouter. This is a
different check than validate_judge.py: that script asks "does the
judge agree with the gold set." This script asks "should the gold
set's own labels be trusted in the first place." A single person's
labels, even written carefully with rationale for the tricky cases,
haven't been independently cross-checked by anything until this runs.

Same rubric as judge.py, imported directly rather than duplicated, so
a fair comparison depends on all three raters (human, GPT-4o, DeepSeek)
being scored against the exact same standard.

Requires OPENROUTER_API_KEY. Writes gold_set_gpt4o.json and
gold_set_deepseek.json alongside the original gold_set.json. Run
compare_labels.py afterward to see where they agree and where they
don't.
"""

from __future__ import annotations

import json
import os

import httpx
from dotenv import load_dotenv

from judge import JUDGE_SYSTEM_PROMPT, RUBRIC_ITEMS, _build_user_prompt, _parse_judge_response

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Confirmed against OpenRouter's own model listing, not guessed at. If
# either ever 404s (OpenRouter's catalog does shift model IDs over
# time), check openrouter.ai/models for the current slug and update
# here, everything downstream just consumes MODEL_SLUGS.
MODEL_SLUGS = {
    "gpt4o": "openai/gpt-4o",
    "deepseek": "deepseek/deepseek-chat",
}

GOLD_SET_PATH = os.path.join(os.path.dirname(__file__), "gold_set.json")


def _call_openrouter(model_slug: str, system_prompt: str, user_prompt: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")

    response = httpx.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model_slug,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def label_turn_with_model(model_slug: str, persona: str, user_message: str, reply_text: str) -> dict:
    user_prompt = _build_user_prompt(persona, user_message, reply_text)
    raw_text = _call_openrouter(model_slug, JUDGE_SYSTEM_PROMPT, user_prompt)
    return _parse_judge_response(raw_text)


def label_gold_set_with_model(model_key: str, gold_set: list[dict]) -> list[dict]:
    model_slug = MODEL_SLUGS[model_key]
    labeled = []

    for turn in gold_set:
        try:
            result = label_turn_with_model(
                model_slug, turn["persona"], turn["user_message"], turn["reply_text"]
            )
            labels = {item: result[item]["passed"] for item in RUBRIC_ITEMS}
            error = None
        except Exception as exc:
            # One failed call shouldn't lose the whole run, record the
            # error against this turn and keep going.
            labels = {item: None for item in RUBRIC_ITEMS}
            error = str(exc)

        labeled.append({"turn_id": turn["turn_id"], "labels": labels, "error": error})

    return labeled


def save_labels(labeled: list[dict], filename: str) -> None:
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(labeled, f, indent=2)


def run() -> None:
    with open(GOLD_SET_PATH, encoding="utf-8") as f:
        gold_set = json.load(f)

    for model_key in MODEL_SLUGS:
        print(f"Labeling {len(gold_set)} turns with {model_key} ({MODEL_SLUGS[model_key]})...")
        labeled = label_gold_set_with_model(model_key, gold_set)
        failures = [t for t in labeled if t["error"]]
        if failures:
            print(f"  {len(failures)} turn(s) failed, see the 'error' field in the output file")
        filename = f"gold_set_{model_key}.json"
        save_labels(labeled, filename)
        print(f"  saved to {filename}")


if __name__ == "__main__":
    run()
