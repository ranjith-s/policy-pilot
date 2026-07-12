"""
test_funnel.py — validation of the guided-funnel policy (no LLM service
needed: fact extraction is faked, relevance uses the keyword fallback).

Run:  python tests/test_funnel.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from funnel import FunnelAgent


class FakeExtractor:
    """Returns canned extraction JSON per call; counts calls so tests can
    prove 'more' costs zero LLM calls."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        return self.replies.pop(0)


def run():
    failures = 0

    def check(cond, label):
        nonlocal failures
        if cond:
            print(f"ok    {label}")
        else:
            failures += 1
            print(f"FAIL  {label}")

    # ---- turn 1: farmer casts a wide net -------------------------------
    llm = FakeExtractor([
        json.dumps({"fields": {"occupation": "farmer", "state": "Tamil Nadu"},
                    "keywords": "farmer"}),
        json.dumps({"fields": {"age": 40, "annual_income": 90000},
                    "keywords": ""}),
    ])
    ag = FunnelAgent(llm, use_semantic=False)   # keyword relevance: deterministic
    r1 = ag.run_turn("I'm a farmer in Tamil Nadu")

    check(ag.profile == {"occupation": "farmer", "state": "Tamil Nadu"},
          "profile saved from extraction")
    check(len(ag.candidates) > 100,
          f"wide net: {len(ag.candidates)} candidate schemes surfaced")
    check("LIKELY MATCHES" in r1["text"], "partials shown to the user")
    check("To narrow this down:" in r1["text"], "follow-up question always asked")
    shown = r1["text"].lower()
    check(any(w in shown for w in ("farmer", "kisan", "agri", "krishi", "farm")),
          "farmer-relevant scheme names in the first page")
    check("swachh bharat" not in shown.split("to narrow this down")[0],
          "no generic filler scheme on page one")
    check(llm.calls == 1, "turn used exactly ONE LLM call")

    # ---- 'more' pages with zero LLM calls ------------------------------
    before = llm.calls
    r2 = ag.run_turn("more")
    check(llm.calls == before, "'more' used ZERO LLM calls")
    page1_names = [l for l in r1["text"].splitlines() if l.strip()[:2].rstrip(".").isdigit()]
    page2_names = [l for l in r2["text"].splitlines() if l.strip()[:2].rstrip(".").isdigit()]
    check(page2_names and not set(page1_names) & set(page2_names),
          "'more' shows new schemes, no repeats")

    # ---- turn 2: narrowing ---------------------------------------------
    n_cand_before = len(ag.candidates)
    r3 = ag.run_turn("I'm 40 and earn about 90000 a year")
    check(ag.profile.get("age") == 40 and ag.profile.get("annual_income") == 90000,
          "follow-up facts merged into profile")
    check(len(ag.candidates) < n_cand_before,
          f"candidates narrowed: {n_cand_before} -> {len(ag.candidates)}")
    check(len(ag.eligible) > 0, f"confirmed eligible grew to {len(ag.eligible)}")

    # ---- empty-start 'more' is graceful --------------------------------
    ag2 = FunnelAgent(FakeExtractor([]), use_semantic=False)
    r4 = ag2.run_turn("more")
    check("Tell me about yourself" in r4["text"], "'more' before any facts is graceful")

    # ---- extraction failure never dead-ends ----------------------------
    ag3 = FunnelAgent(FakeExtractor(["not json at all", "still not json"]),
                      use_semantic=False)
    r5 = ag3.run_turn("gibberish input")
    check(r5["type"] == "answer" and r5["text"],
          "extraction failure still produces an answer")

    print(f"\n{'ALL TESTS PASSED' if failures == 0 else f'{failures} FAILURES'}")
    return failures


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
