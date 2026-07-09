"""
backend/suggestions/chips.py
------------------------------
Chip detection for the demo, trimmed to two detectors: phone numbers and
URLs found in the AI's own reply text.

A fuller production suggestion system would cover more detector types
tied to persistent, database-backed features this demo doesn't have:
emergency contacts, medication reminders, location sharing, and similar
profile-driven signals. Porting those here would mean either faking
data for features that don't exist, or reimplementing product surfaces
that don't belong in a portfolio demo.

What's left after removing everything tied to features this build
doesn't have is exactly the two detectors requested: scanning the LLM's
reply text directly for a phone number or a URL, with no dependency on
enrichment, contacts, or profile data. Both are original regex written
fresh for English text.

Chip shape:
    {
        "id": str,
        "label": str,
        "action": "phone_call" | "external_url",
        "target": str,
        "metadata": dict,
    }

Both chip types render as tappable pills per the UI spec, action
determines the tap behavior on the frontend (tel: link vs. opening the
URL in a new tab).
"""

import re

_PHONE_RE = re.compile(r'\b(\d{3}[-.\s]\d{3}[-.\s]\d{4})\b')

_URL_RE = re.compile(
    r'https?://[^\s"\')\]]+'
    r'|(?:www\.)[a-zA-Z0-9][a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s"\')\]]*)?'
    r'|\b[a-zA-Z0-9][a-zA-Z0-9-]{2,}'
    r'\.(?:com|org|net|edu|gov|io|app|info|co)'
    r'(?:/[^\s"\')\]]*)?'
)


def detect_phone_chip(reply_text: str) -> dict | None:
    """
    Scan the AI's reply for a North American phone number
    (XXX-XXX-XXXX, XXX.XXX.XXXX, or XXX XXX XXXX) and return a tappable
    call chip if one is found, else None.
    """
    match = _PHONE_RE.search(reply_text)
    if not match:
        return None

    phone = match.group(1)
    return {
        "id": "call_business_reply",
        "label": f"Call {phone}",
        "action": "phone_call",
        "target": phone,
        "metadata": {},
    }


def detect_url_chip(reply_text: str, existing_targets: set[str] | None = None) -> dict | None:
    """
    Scan the AI's reply for a web address (full URL, www-prefixed, or a
    bare domain with a common TLD) and return a tappable open-link chip
    if one is found and isn't already present in existing_targets.

    Common TLDs only, to avoid false positives on things like "Dr. Lee"
    or a sentence-ending period after a number.
    """
    found = _URL_RE.search(reply_text)
    if not found:
        return None

    raw = found.group(0).rstrip('.,!?\'")]')
    launch_url = raw if raw.startswith("http") else f"https://{raw}"

    try:
        netloc = launch_url.split("://")[1].split("/")[0]
    except Exception:
        netloc = raw.split("/")[0]

    if not netloc:
        return None

    if existing_targets and launch_url.rstrip("/") in existing_targets:
        return None

    return {
        "id": "open_url_reply",
        "label": f"Open {netloc}",
        "action": "external_url",
        "target": launch_url,
        "metadata": {},
    }


def detect_chips(reply_text: str) -> list[dict]:
    """
    Run both detectors against a reply and return whatever chips were
    found, phone first: an actionable phone number outranks a link.
    """
    chips: list[dict] = []

    phone_chip = detect_phone_chip(reply_text)
    if phone_chip:
        chips.append(phone_chip)

    url_chip = detect_url_chip(
        reply_text, existing_targets={c["target"] for c in chips if c["action"] == "external_url"}
    )
    if url_chip:
        chips.append(url_chip)

    return chips
