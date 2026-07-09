"""
backend/eval/compare_labels.py
---------------------------------
Compares the gold set's original human labels against the two
independent model labelings produced by label_with_models.py. Where
all three agree, that's real confirmation the label is trustworthy.
Where they don't, that's a specific, concrete thing worth a second
look, not a vague sense that the gold set might need review.

Run label_with_models.py first, this script only reads what that one
produces.
"""

from __future__ import annotations

import json
import os

from judge import RUBRIC_ITEMS

EVAL_DIR = os.path.dirname(__file__)


def _load(filename: str) -> list[dict]:
    path = os.path.join(EVAL_DIR, filename)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare(
    human_set: list[dict] | None = None,
    gpt4o_set: list[dict] | None = None,
    deepseek_set: list[dict] | None = None,
) -> dict:
    """
    Compares three label sources per applicable item. Arguments are
    optional and load from the standard files when omitted, injectable
    for testing.
    """
    human_set = human_set if human_set is not None else _load("gold_set.json")
    gpt4o_by_id = {t["turn_id"]: t["labels"] for t in (gpt4o_set if gpt4o_set is not None else _load("gold_set_gpt4o.json"))}
    deepseek_by_id = {t["turn_id"]: t["labels"] for t in (deepseek_set if deepseek_set is not None else _load("gold_set_deepseek.json"))}

    total_applicable = 0
    full_agreement = 0
    disagreements = []

    for turn in human_set:
        turn_id = turn["turn_id"]
        for item in RUBRIC_ITEMS:
            human_label = turn["labels"].get(item)
            if human_label is None:
                continue  # not applicable per the human label, nothing to cross-check

            total_applicable += 1
            gpt4o_label = gpt4o_by_id.get(turn_id, {}).get(item)
            deepseek_label = deepseek_by_id.get(turn_id, {}).get(item)

            if gpt4o_label == human_label and deepseek_label == human_label:
                full_agreement += 1
            else:
                disagreements.append({
                    "turn_id": turn_id,
                    "item": item,
                    "human": human_label,
                    "gpt4o": gpt4o_label,
                    "deepseek": deepseek_label,
                })

    return {
        "total_applicable": total_applicable,
        "full_agreement": full_agreement,
        "disagreements": disagreements,
    }


def print_report(result: dict) -> None:
    total = result["total_applicable"]
    agree = result["full_agreement"]
    print("=== Cross-check: human label vs GPT-4o vs DeepSeek V3 ===")
    print(f"Applicable label instances: {total}")
    rate = f"{agree / total:.1%}" if total else "n/a"
    print(f"Full 3-way agreement: {agree} ({rate})")
    print(f"Disagreements needing review: {len(result['disagreements'])}")
    if result["disagreements"]:
        print()
        print("--- Worth a second look ---")
        for d in result["disagreements"]:
            print(f"{d['turn_id']} / {d['item']}: human={d['human']}, gpt4o={d['gpt4o']}, deepseek={d['deepseek']}")


if __name__ == "__main__":
    print_report(compare())
