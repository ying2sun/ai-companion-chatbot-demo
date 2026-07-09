"""
backend/suggestions/units.py
------------------------------
Detects a measurement mentioned in the AI's reply (temperature, weight,
distance, height) and returns a non-tappable info chip showing the
converted value. All arithmetic runs in plain Python, never through the
LLM, an LLM asked to do unit math is unreliable at it in a way a
one-line formula is not, so this keeps that math out of the model's
hands entirely.

This is the untappable chip style built earlier and left unused, this
module is what finally gives it something to render.
"""

import re
import logging
from typing import Callable

logger = logging.getLogger(__name__)

MAX_UNIT_CHIPS = 2  # cap per response, avoids flooding the chip row

_NUM = r"(\d+(?:\.\d+)?)"

# -----------------------------------------------------------------------------
# Conversion functions
# -----------------------------------------------------------------------------

def _c_to_f(v: float) -> str:
    return f"{round(v * 9 / 5 + 32)}\u00b0F"

def _f_to_c(v: float) -> str:
    result = round((v - 32) * 5 / 9, 1)
    return f"{int(result) if result == int(result) else result}\u00b0C"

def _kg_to_lb(v: float) -> str:
    return f"{round(v * 2.20462, 1)} lb"

def _lb_to_kg(v: float) -> str:
    return f"{round(v / 2.20462, 1)} kg"

def _km_to_miles(v: float) -> str:
    return f"{round(v / 1.60934, 1)} miles"

def _miles_to_km(v: float) -> str:
    return f"{round(v * 1.60934, 1)} km"

def _cm_to_ft_in(v: float) -> str:
    total_in = v / 2.54
    ft = int(total_in // 12)
    inch = round(total_in % 12)
    if inch == 12:
        ft += 1
        inch = 0
    return f"{ft}ft {inch}in"


# -----------------------------------------------------------------------------
# Detection patterns, English phrasing. Each entry: (pattern, from_unit, fn)
# -----------------------------------------------------------------------------

_DETECTORS: list[tuple[re.Pattern, str, Callable[[float], str]]] = [
    (re.compile(_NUM + r"\s*(?:\u00b0C|degrees?\s*Celsius)", re.IGNORECASE), "C", _c_to_f),
    (re.compile(_NUM + r"\s*(?:\u00b0F|degrees?\s*Fahrenheit)", re.IGNORECASE), "F", _f_to_c),
    (re.compile(_NUM + r"\s*(?:kg|kilograms?)\b", re.IGNORECASE), "kg", _kg_to_lb),
    (re.compile(_NUM + r"\s*(?:lbs?|pounds?)\b", re.IGNORECASE), "lb", _lb_to_kg),
    (re.compile(_NUM + r"\s*(?:km|kilomet(?:er|re)s?)\b", re.IGNORECASE), "km", _km_to_miles),
    (re.compile(_NUM + r"\s*(?:miles?|mi\b)", re.IGNORECASE), "miles", _miles_to_km),
    (re.compile(_NUM + r"\s*(?:cm|centimet(?:er|re)s?)\b", re.IGNORECASE), "cm", _cm_to_ft_in),
]

# Suppress a conversion chip when the reply already states the converted
# figure in its target unit, an LLM will sometimes write both units inline
# ("400F, about 204C"), and a chip in that case is redundant and can look
# contradictory once rounding is involved. Tolerance is keyed by the target
# unit and sized to absorb ordinary rounding.
_TARGET_UNIT = {"C": "F", "F": "C", "kg": "lb", "lb": "kg", "km": "miles", "miles": "km", "cm": None}
_COUNTERPART_TOL = {"C": 1.0, "F": 2.0, "kg": 0.5, "lb": 1.0, "km": 0.3, "miles": 0.3}


def _lead_num(s: str):
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def _present_values(reply_text: str) -> dict:
    present: dict = {}
    for pattern, from_unit, _fn in _DETECTORS:
        for m in pattern.finditer(reply_text):
            try:
                present.setdefault(from_unit, []).append(float(m.group(1)))
            except (IndexError, ValueError):
                continue
    return present


def detect_unit_chips(reply_text: str) -> list[dict]:
    """
    Scan reply_text for a recognized measurement and return up to
    MAX_UNIT_CHIPS chip dicts with action="unit_info". The same
    (value, unit) pair is never emitted twice.
    """
    chips: list[dict] = []
    seen: set[str] = set()
    present = _present_values(reply_text)

    for pattern, from_unit, converter in _DETECTORS:
        if len(chips) >= MAX_UNIT_CHIPS:
            break
        for match in pattern.finditer(reply_text):
            if len(chips) >= MAX_UNIT_CHIPS:
                break
            try:
                value = float(match.group(1))
            except (IndexError, ValueError):
                continue

            dedup_key = f"{value}{from_unit}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            try:
                converted = converter(value)
            except Exception as exc:
                logger.warning("Unit conversion failed | value=%s | from=%s | error=%s", value, from_unit, exc)
                continue

            target_unit = _TARGET_UNIT.get(from_unit)
            converted_num = _lead_num(converted)
            tol = _COUNTERPART_TOL.get(target_unit) if target_unit else None
            if (
                tol is not None
                and converted_num is not None
                and any(abs(v - converted_num) <= tol for v in present.get(target_unit, []))
            ):
                continue

            display = int(value) if value == int(value) else value
            unit_symbol = f"\u00b0{from_unit}" if from_unit in ("C", "F") else f" {from_unit}"
            label = f"{display}{unit_symbol} = {converted}"

            chips.append({
                "id": f"unit_info_{len(chips)}",
                "label": label,
                "action": "unit_info",
                "target": None,
                "metadata": {"from_value": value, "from_unit": from_unit, "converted": converted},
            })

    return chips
