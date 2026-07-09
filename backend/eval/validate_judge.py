"""
backend/eval/validate_judge.py
--------------------------------
Runs the judge against every applicable turn in the gold set, compares
its labels to the human ones already recorded there, and computes
Cohen's kappa per rubric item, the same validation approach the
README's Evaluation section explains conceptually. This is where that
explanation becomes a real, runnable number.

Requires a real ANTHROPIC_API_KEY to run against actual judge output.
The kappa math itself (compute_kappa) is verified independently below
against hand-calculated cases, so its correctness doesn't depend on
having a key at all.
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

from judge import judge_turn, RUBRIC_ITEMS

load_dotenv()

GOLD_SET_PATH = os.path.join(os.path.dirname(__file__), "gold_set.json")

KAPPA_GATE_THRESHOLD = 0.6  # matches the threshold the README commits to


def compute_kappa(human_labels: list[bool], judge_labels: list[bool]) -> float | None:
    """
    Cohen's kappa between two raters on the same set of binary items.

    Returns None when kappa is undefined, both raters showed zero
    variation (agreed with themselves on every item), which makes the
    denominator zero. That's the exact case the README's Evaluation
    section documents: it means there was nothing to measure agreement
    against, not that the judge is unreliable.
    """
    n = len(human_labels)
    if n == 0:
        return None

    p_observed = sum(1 for h, j in zip(human_labels, judge_labels) if h == j) / n

    human_true_rate = sum(human_labels) / n
    judge_true_rate = sum(judge_labels) / n
    p_expected = (human_true_rate * judge_true_rate) + (
        (1 - human_true_rate) * (1 - judge_true_rate)
    )

    if p_expected == 1.0:
        return None

    return (p_observed - p_expected) / (1 - p_expected)


def run_validation() -> dict:
    with open(GOLD_SET_PATH, encoding="utf-8") as f:
        gold_set = json.load(f)

    per_item = {item: {"human": [], "judge": []} for item in RUBRIC_ITEMS}

    for turn in gold_set:
        judge_result = judge_turn(turn["persona"], turn["user_message"], turn["reply_text"])
        for item in RUBRIC_ITEMS:
            human_label = turn["labels"].get(item)
            if human_label is None:
                continue  # human marked this item not applicable, skip it
            judge_label = judge_result[item]["passed"]
            per_item[item]["human"].append(bool(human_label))
            per_item[item]["judge"].append(bool(judge_label) if judge_label is not None else False)

    summary = {}
    for item in RUBRIC_ITEMS:
        h, j = per_item[item]["human"], per_item[item]["judge"]
        n = len(h)
        kappa = compute_kappa(h, j)
        raw_agreement = sum(1 for a, b in zip(h, j) if a == b) / n if n else None
        summary[item] = {"n": n, "raw_agreement": raw_agreement, "kappa": kappa}

    return summary


def print_summary(summary: dict) -> None:
    print("=== Judge validation against gold set ===")
    for item, stats in summary.items():
        kappa = stats["kappa"]
        kappa_str = f"{kappa:.3f}" if kappa is not None else "undefined (no variation to measure)"
        gate = ""
        if kappa is not None:
            gate = "PASS" if kappa >= KAPPA_GATE_THRESHOLD else "BELOW THRESHOLD"
        agreement_str = f"{stats['raw_agreement']:.1%}" if stats["raw_agreement"] is not None else "n/a"
        print(f"{item}: n={stats['n']}, raw_agreement={agreement_str}, kappa={kappa_str} {gate}")


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        # Verifies compute_kappa's math against hand-calculated cases,
        # no API key, no gold set, no judge call needed for this part.
        cases_ok = True

        # Case 1: real, hand-calculated example.
        # human = [T,T,T,T,T,T,T,F,F,F] (7 true / 3 false)
        # judge = [T,T,T,T,T,T,F,F,F,F] (6 true / 4 false)
        # 9/10 raw agreement, p_expected = 0.7*0.6 + 0.3*0.4 = 0.54
        # kappa = (0.9 - 0.54) / (1 - 0.54) = 0.36 / 0.46 = 0.7826...
        h1 = [True] * 7 + [False] * 3
        j1 = [True] * 6 + [False, False, False, False]
        k1 = compute_kappa(h1, j1)
        expected1 = 0.36 / 0.46
        ok1 = k1 is not None and abs(k1 - expected1) < 1e-6
        cases_ok = cases_ok and ok1
        print(f"[{'OK' if ok1 else 'MISMATCH'}] real-signal case: kappa={k1:.4f}, expected={expected1:.4f}")

        # Case 2: high raw agreement, near-zero kappa. Human and judge
        # each say True on 19/20, but disagree on two DIFFERENT items,
        # so raw agreement is 90% while kappa is close to zero, exactly
        # the "raw agreement is misleading" case the README explains.
        h2 = [True] * 20
        h2[4] = False
        j2 = [True] * 20
        j2[9] = False
        k2 = compute_kappa(h2, j2)
        ok2 = k2 is not None and abs(k2) < 0.1
        cases_ok = cases_ok and ok2
        print(f"[{'OK' if ok2 else 'MISMATCH'}] high-agreement-low-kappa case: kappa={k2:.4f} (expected near 0)")

        # Case 3: undefined kappa, both raters agree with themselves on
        # every single item (all True), zero variation, zero denominator.
        h3 = [True] * 15
        j3 = [True] * 15
        k3 = compute_kappa(h3, j3)
        ok3 = k3 is None
        cases_ok = cases_ok and ok3
        print(f"[{'OK' if ok3 else 'MISMATCH'}] undefined case: kappa={k3} (expected None)")

        # Case 4: perfect agreement with real variation, kappa should be
        # exactly 1.0.
        h4 = [True, False, True, False, True, False]
        j4 = [True, False, True, False, True, False]
        k4 = compute_kappa(h4, j4)
        ok4 = k4 is not None and abs(k4 - 1.0) < 1e-9
        cases_ok = cases_ok and ok4
        print(f"[{'OK' if ok4 else 'MISMATCH'}] perfect-agreement case: kappa={k4}")

        print()
        print("ALL KAPPA MATH CASES PASSED" if cases_ok else "SOME CASES FAILED")
    else:
        result = run_validation()
        print_summary(result)
