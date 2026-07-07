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

_semantic_index = None
_semantic_tried = False
_rules_ids_cache = None


def _rules_ids():
    """Scheme ids that have annotated rows in scheme_rules.csv."""
    global _rules_ids_cache
    if _rules_ids_cache is None:
        import csv
        with open(DATA_DIR / "scheme_rules.csv", newline="", encoding="utf-8") as f:
            _rules_ids_cache = {row["id"] for row in csv.DictReader(f)}
    return _rules_ids_cache


def _get_semantic_index():
    """Lazy-load the semantic index; never raises (fallback = keyword)."""
    global _semantic_index, _semantic_tried
    if not _semantic_tried:
        _semantic_tried = True
        try:
            from embeddings import SemanticIndex
            if SemanticIndex.available():
                _semantic_index = SemanticIndex()
        except Exception:
            _semantic_index = None
    return _semantic_index


def _metadata_filter(state=None, category=None):
    """Tier-1 hard filter. Returns list of candidate docs."""
    out = []
    for d in _corpus().values():
        if state and state.lower() not in [s.lower() for s in d["states"]] \
                and "all" not in [s.lower() for s in d["states"]]:
            continue
        if category and category.lower() not in " ".join(d["categories"]).lower():
            continue
        out.append(d)
    return out


def search_schemes(query="", state=None, category=None, max_results=5, **_):
    """Hybrid retrieval: metadata hard-filter -> semantic (if index built)
    blended with keyword score. Falls back to pure keyword when the
    embedding index or Ollama is unavailable."""
    candidates = _metadata_filter(state, category)
    cand_ids = {d["id"] for d in candidates}
    by_id = {d["id"]: d for d in candidates}

    query_words = [w for w in (query or "").lower().split() if len(w) > 2]

    def kw_score(d):
        return sum(d["search_text"].count(w) for w in query_words)

    mode = "keyword"
    ranked = []

    index = _get_semantic_index() if query else None
    if index is not None:
        try:
            sem = index.query(query, top_k=max(max_results * 3, 10),
                              restrict_ids=cand_ids)
            if sem:
                mode = "hybrid"
                max_kw = max((kw_score(by_id[sid]) for sid, _ in sem), default=0) or 1
                ranked = sorted(
                    ((0.7 * s + 0.3 * (kw_score(by_id[sid]) / max_kw), by_id[sid])
                     for sid, s in sem),
                    key=lambda x: -x[0],
                )
        except Exception:
            ranked = []          # embedding call failed mid-flight -> keyword

    if not ranked:               # keyword fallback (or empty query = browse)
        ranked = sorted(
            ((kw_score(d), d) for d in candidates
             if kw_score(d) > 0 or not query_words),
            key=lambda x: -x[0],
        )

    return {
        "retrieval_mode": mode,
        "matches": [
            {
                "scheme_id": d["id"],
                "scheme_name": d["scheme_name"],
                "level": d["level"],
                "states": d["states"],
                "brief": d["brief_description"][:200],
                # can the engine actually verdict this scheme?
                "rules_available": d["id"] in _rules_ids(),
            }
            for _, d in ranked[:max_results]
        ],
        "total_found": len(ranked),
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
   Each match includes "rules_available": true means run_eligibility_check can
   give a verdict for it; false means you may only DESCRIBE the scheme and point
   the user to its official page — never guess its eligibility.

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
        # Validate required parameters
        if "field" not in action_input or "value" not in action_input:
            return {"error": f"update_profile requires 'field' and 'value', got {list(action_input.keys())}"}
        return update_profile(profile_store, **action_input)
    return {"error": f"unknown tool '{action}'"}
