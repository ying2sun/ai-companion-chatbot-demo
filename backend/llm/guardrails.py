"""
backend/llm/guardrails.py
--------------------------
Rule-based guardrail that runs on every message BEFORE the LLM reply is
returned.

This is an original, English-language reimplementation of the production
guardrail's design: a deterministic regex pre-check as a fast first layer,
with the system prompt's own safety instructions as a second line of
defense. No production trigger phrases or override wording are carried
over, everything below is written fresh for this demo. The Simplified to
Traditional Chinese normalization step in the production version is
dropped entirely, it has no equivalent need in an English-only build.

Why keyword matching instead of a second LLM call
---------------------------------------------------
A second LLM call to judge safety adds real latency and real cost at
scale. For a threat model centered on medical advice and crisis language,
regex on the user's message is:
  - Fast: well under a millisecond, no network call
  - Deterministic: the same input always gives the same result, easy to
    test
  - Free: no token cost
  - Auditable: every rule is readable, there's nothing hidden in a model's
    judgment

The tradeoff is brittleness: a novel phrasing that doesn't match any
pattern slips through. That's why the system prompt in prompts.py also
carries a safety instruction as backup. Neither layer alone is sufficient,
together they cover this demo's scope.

Trigger order matters
-----------------------
Crisis is checked before medical. A message that reads as both ("I don't
want to be here anymore, can I just take all my pills at once") has to be
caught as a crisis case, not filed as a medication question. The crisis
override is written to keep the person talking, the medical override
redirects to a doctor. Getting this order backwards sends the wrong signal
in the highest-stakes case.

The guardrail runs on the user's message only
-------------------------------------------------
This checks input, not the model's output. Checking output creates a race
condition: the model could already have generated something unsafe before
the check runs. Checking input means the override happens before the LLM
generates anything at all. When the guardrail fires, the LLM reply (if any
was even started) is discarded and replaced with the override text.

A note on this demo's trigger list
-------------------------------------
These patterns are a small, illustrative set, not a clinically reviewed
production list. If you use this beyond a portfolio demo, treat this file
as a starting point that needs real review, the same way the production
list would have gone through iteration based on real conversations.
"""

import re

# Crisis: expressions of suicidal ideation or a wish to die. Matched
# conservatively to avoid firing on figurative speech ("I could just die of
# embarrassment"). These represent fairly direct expressions, not every
# possible phrasing.
CRISIS_TRIGGERS = [
    r"don'?t want to (live|be here|go on)",
    r"want(ed)? to die",
    r"better off dead",
    r"no (point|reason) (in )?living",
    r"end(ing)? (it all|my life)",
    r"kill(ing)? myself",
    r"suicid",
    r"can'?t go on",
    r"not worth living",
    r"wish I (was|were) dead",
    r"tired of (being alive|living)",
]

# Medical: dosage questions, medication changes, side-effect questions,
# stopping medication. The wildcards handle natural variation in how the
# question gets asked ("can I take an extra one of my blood pressure
# pills" vs. "take more medicine").
MEDICAL_TRIGGERS = [
    r"how many (pills|tablets)",
    r"take (more|extra) (of )?(my|the) (medicine|medication|pills?)",
    r"double (my|the) dose",
    r"dosage",
    r"side effects?",
    r"drug interactions?",
    r"stop taking (my|the) (medicine|medication)",
    r"skip (my|a|the) (dose|medication)",
    r"can I take .* (medicine|medication|pills?)",
    r"mix .* (medicine|medication|pills?)",
]

# Design principle carried over from production: the redirect should feel
# caring, not like a legal disclaimer.
#
# Crisis override keeps the person talking and emphasizes they're not
# alone. It deliberately does not escalate to a family member immediately,
# that kind of escalation belongs in a real product with a real safety
# team behind it, not a portfolio demo. It stays present and invites the
# person to share more, a holding response, not a risk assessment.
#
# Medical override is warm, redirects to a doctor, and offers a concrete
# next step. It intentionally says nothing medical at all.

CRISIS_OVERRIDE = (
    "I'm worried about you hearing this. You're not alone, I'm right here "
    "with you. Can you tell me a little about what's been happening?"
)

MEDICAL_OVERRIDE = (
    "Medication questions are best answered by your doctor. Would you "
    "like help thinking about how to ask a family member to take you?"
)


def check_guardrails(user_message: str, llm_response: str = "") -> dict:
    """
    Check user_message against crisis and medical trigger patterns.

    Returns a dict shaped like:
        {
            "triggered": bool,
            "reason": str | None,
            "override_response": str | None,
        }

    The caller (the /chat endpoint) should:
      1. If triggered, replace the reply text with override_response.
      2. If triggered, drop any suggestion chips for that turn.
      3. Always log guardrail_triggered and guardrail_reason, even when
         not triggered, it's useful for a demo debug panel.

    Parameters
    ----------
    user_message : the person's raw input text (post-STT or typed directly)
    llm_response  : accepted for interface parity with a possible future
                    output-side check, not inspected here
    """
    for pattern in CRISIS_TRIGGERS:
        if re.search(pattern, user_message, re.IGNORECASE):
            return {
                "triggered": True,
                "reason": "crisis_language_detected",
                "override_response": CRISIS_OVERRIDE,
            }

    for pattern in MEDICAL_TRIGGERS:
        if re.search(pattern, user_message, re.IGNORECASE):
            return {
                "triggered": True,
                "reason": "medical_advice_request",
                "override_response": MEDICAL_OVERRIDE,
            }

    return {"triggered": False, "reason": None, "override_response": None}
