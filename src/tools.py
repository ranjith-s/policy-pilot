"""
tools.py — The agent's toolbox.

Each tool is a plain Python function. The agent (LLM) chooses which one to
call each turn; this module executes it and returns an observation dict.
"""

import json
from pathlib import Path

from engine import check_eligibility, get_next_question, KNOWN_FIELDS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_corpus_cache = None


def _corpus():
    global _corpus_cache
    if _corpus_cache is None:
        with open(DATA_DIR / "rag_corpus.json", encoding="utf-8") as f:
            _corpus_cache = {d["id"]: d for d in json.load(f)}
    return _corpus_cache


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def search_schemes(query="", state=None, category=None, max_results=5, **_):
    """Keyword + metadata search over the corpus (the discovery tool)."""
    query_words = [w for w in (query or "").lower().split() if len(w) > 2]
    scored = []
    for d in _corpus().values():
        if state and state.lower() not in [s.lower() for s in d["states"]] \
                and "all" not in [s.lower() for s in d["states"]]:
            continue
        if category and category.lower() not in " ".join(d["categories"]).lower():
            continue
        score = sum(d["search_text"].count(w) for w in query_words)
        if score > 0 or not query_words:
            scored.append((score, d))
    scored.sort(key=lambda x: -x[0])
    return {
        "matches": [
            {
                "scheme_id": d["id"],
                "scheme_name": d["scheme_name"],
                "level": d["level"],
                "states": d["states"],
                "brief": d["brief_description"][:200],
            }
            for _, d in scored[:max_results]
        ],
        "total_found": len(scored),
    }


def run_eligibility_check(profile, **_):
    """Deterministic engine + next-question suggestion. THE source of truth."""
    results = check_eligibility(profile)
    nq = get_next_question(profile, results)
    return {
        "results": results,
        "suggested_next_question": nq,
        "summary": {
            "eligible": [r["scheme_name"] for r in results if r["status"] == "eligible"],
            "partial": [r["scheme_name"] for r in results if r["status"] == "partial"],
            "not_eligible": [r["scheme_name"] for r in results if r["status"] == "not_eligible"],
        },
    }


def get_scheme_details(scheme_id, **_):
    """Full details for one scheme (benefits, how to apply)."""
    d = _corpus().get(scheme_id)
    if not d:
        return {"error": f"unknown scheme_id '{scheme_id}'"}
    return {
        "scheme_name": d["scheme_name"],
        "benefits": d["benefits"][:800],
        "eligibility_text": d["eligibility_text"][:800],
        "application": d["application"],
        "documents_required": d["documents_required"],
    }


def update_profile(profile_store, field, value, **_):
    """Write a fact into the session profile (agent memory)."""
    if field not in KNOWN_FIELDS:
        return {"error": f"unknown field '{field}'. Known: {KNOWN_FIELDS}"}
    profile_store[field] = value
    return {"ok": True, "profile": dict(profile_store)}


# ask_user and final_answer are handled by the loop itself (they end the turn),
# but they are declared here so the schema shown to the LLM is complete.

TOOL_SCHEMAS = """
Available tools (choose exactly one per step):

1. search_schemes — find candidate schemes by keywords/filters.
   input: {"query": "farmer loan", "state": "Delhi", "category": "Education"}
   (all inputs optional; use the user's own words as query)

2. run_eligibility_check — check the current profile against all scheme rules.
   input: {} (uses the session profile automatically)
   This is the ONLY source of eligibility verdicts.

3. get_scheme_details — benefits + application steps for one scheme.
   input: {"scheme_id": "sui"}

4. update_profile — save a fact the user just told you.
   input: {"field": "age", "value": 34}
   fields: age, annual_income, gender, category, occupation, state,
           marital_status, has_bank_account, land_owner

5. ask_user — ask the user ONE question when required info is missing.
   input: {"question": "What is your annual income?"}

6. final_answer — end the turn with your answer to the user.
   input: {"answer": "..."}
"""


def dispatch(action, action_input, profile_store):
    """Execute a tool by name. Returns an observation dict."""
    if action == "search_schemes":
        return search_schemes(**action_input)
    if action == "run_eligibility_check":
        return run_eligibility_check(profile=dict(profile_store))
    if action == "get_scheme_details":
        return get_scheme_details(**action_input)
    if action == "update_profile":
        return update_profile(profile_store, **action_input)
    return {"error": f"unknown tool '{action}'"}
