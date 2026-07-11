"""
test_engine.py — persona-based validation of the deterministic engine.

Run:  python tests/test_engine.py
(no pytest dependency needed, though it's pytest-compatible)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from engine import check_eligibility, get_next_question

PERSONAS = [
    {
        "name": "SC woman entrepreneur, 30, has bank account",
        "profile": {"age": 30, "gender": "female", "category": "SC",
                    "occupation": "entrepreneur", "has_bank_account": "yes",
                    "state": "Tamil Nadu", "annual_income": 300000,
                    "marital_status": "married"},
        "expect": {"sui": "eligible", "pmsby": "eligible",
                   "sfava": "not_eligible", "famdpwog": "not_eligible",
                   "rgisfm": "not_eligible"},
    },
    {
        "name": "General-category male entrepreneur, 40",
        "profile": {"age": 40, "gender": "male", "category": "General",
                    "occupation": "entrepreneur", "has_bank_account": "yes",
                    "state": "Delhi", "annual_income": 500000,
                    "marital_status": "married"},
        "expect": {"sui": "not_eligible", "pmsby": "eligible"},
    },
    {
        "name": "Veteran artist, 65, low income",
        "profile": {"age": 65, "gender": "male", "occupation": "artist",
                    "annual_income": 40000, "has_bank_account": "yes",
                    "state": "Kerala", "category": "General",
                    "marital_status": "married"},
        "expect": {"sfava": "eligible", "pmsby": "eligible",
                   "rgisfm": "not_eligible"},
    },
    {
        "name": "Widow in Delhi, income 80k",
        "profile": {"age": 45, "gender": "female", "marital_status": "widow",
                    "state": "Delhi", "annual_income": 80000,
                    "has_bank_account": "yes", "occupation": "homemaker",
                    "category": "General"},
        "expect": {"famdpwog": "eligible", "pmsby": "eligible",
                   "sui": "not_eligible"},
    },
    {
        "name": "75-year-old (too old for PMSBY)",
        "profile": {"age": 75, "gender": "male", "occupation": "retired",
                    "has_bank_account": "yes", "state": "Punjab",
                    "annual_income": 200000, "category": "General",
                    "marital_status": "married"},
        "expect": {"pmsby": "not_eligible"},
    },
    {
        "name": "Faculty member, 45",
        "profile": {"age": 45, "gender": "female", "occupation": "faculty",
                    "has_bank_account": "yes", "state": "Karnataka",
                    "annual_income": 1200000, "category": "General",
                    "marital_status": "married"},
        "expect": {"rgisfm": "eligible", "pmsby": "eligible"},
    },
    {
        "name": "Partial: farmer with only occupation+state known",
        "profile": {"occupation": "farmer", "state": "Tamil Nadu"},
        # nothing should crash; PMSBY should be partial (age+bank unknown)
        "expect": {"pmsby": "partial"},
    },
]


# personas against LLM-extracted rules (present after extract_rules.py merge);
# skipped gracefully if the extracted schemes aren't in scheme_rules.csv yet
LLM_PERSONAS = [
    {
        "name": "Widow 45 in J&K, BPL (LLM-extracted widow pension)",
        "profile": {"age": 45, "gender": "female", "marital_status": "widow",
                    "category": "BPL", "state": "Jammu and Kashmir",
                    "annual_income": 30000, "occupation": "homemaker",
                    "has_bank_account": "yes"},
        "expect": {"ignwpsjak": "eligible"},
    },
    {
        "name": "Male 45 in J&K (must fail widow pension)",
        "profile": {"age": 45, "gender": "male", "marital_status": "married",
                    "category": "BPL", "state": "Jammu and Kashmir",
                    "annual_income": 30000, "occupation": "farmer",
                    "has_bank_account": "yes"},
        "expect": {"ignwpsjak": "not_eligible"},
    },
    {
        "name": "BPL 30yo in Puducherry, income 50k (LLM-extracted RGSSS)",
        "profile": {"age": 30, "gender": "male", "category": "BPL",
                    "state": "Puducherry", "annual_income": 50000,
                    "occupation": "farmer", "marital_status": "married",
                    "has_bank_account": "yes"},
        "expect": {"rgssspf-2012": "eligible"},
    },
]


def run():
    failures = 0
    known_ids = {r["scheme_id"] for r in check_eligibility({})}
    personas = list(PERSONAS)
    for p in LLM_PERSONAS:
        if set(p["expect"]) <= known_ids:
            personas.append(p)
        else:
            print(f"skip  [{p['name']}] — scheme not in rules CSV yet")
    for p in personas:
        results = {r["scheme_id"]: r for r in check_eligibility(p["profile"])}
        for scheme_id, expected in p["expect"].items():
            got = results[scheme_id]["status"]
            ok = got == expected
            if not ok:
                failures += 1
                print(f"FAIL  [{p['name']}] {scheme_id}: expected {expected}, got {got}")
                print(f"      reasons: {results[scheme_id]['reasons']}")
                print(f"      missing: {results[scheme_id]['missing_fields']}")
            else:
                print(f"ok    [{p['name']}] {scheme_id}: {got}")

    # next-question logic
    partial_profile = {"occupation": "farmer", "state": "Tamil Nadu"}
    nq = get_next_question(partial_profile, check_eligibility(partial_profile))
    assert nq is not None, "expected a follow-up question for a sparse profile"
    print(f"ok    next-question for sparse profile -> asks about '{nq['field']}'")

    complete = PERSONAS[0]["profile"]
    nq2 = get_next_question(complete, check_eligibility(complete))
    assert nq2 is None, "complete profile should not trigger questions"
    print("ok    complete profile -> no follow-up question")

    print(f"\n{'ALL TESTS PASSED' if failures == 0 else f'{failures} FAILURES'}")
    return failures


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
