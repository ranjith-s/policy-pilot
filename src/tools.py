"""
tools.py — The agent's toolbox.

Each tool is a plain Python function. The agent (LLM) chooses which one to
call each turn; this module executes it and returns an observation dict.
"""

import json
from pathlib import Path

from engine import check_eligibility, get_next_question, KNOWN_FIELDS, _is_annotated

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
            _rules_ids_cache = {row["id"] for row in csv.DictReader(f) if _is_annotated(row)}
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
    # 'India' / 'all' is not a state — treat as no filter
    if state and state.strip().lower() in ("india", "all", "all india", "any"):
        state = None
    out = []
    for d in _corpus().values():
        if state and state.lower() not in [s.lower() for s in d["states"]] \
                and "all" not in [s.lower() for s in d["states"]]:
            continue
        if category and category.lower() not in " ".join(d["categories"]).lower():
            continue
        out.append(d)
    return out


def search_schemes(query="", state=None, category=None, max_results=4, **_):
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

    result_hint = None
    if not ranked and (state or category):
        result_hint = ("No matches with these filters. Retry with query words "
                       "only — no state/category filter.")

    return {
        "retrieval_mode": mode,
        **({"hint": result_hint} if result_hint else {}),
        "matches": [
            {
                "scheme_id": d["id"],
                "scheme_name": d["scheme_name"],
                "level": d["level"],
                "states": d["states"],
                "brief": d["brief_description"][:120],
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


def compact_engine_obs(obs, max_eligible=10, max_partial=5):
    """What the LLM sees after run_eligibility_check. With thousands of
    annotated schemes the full result would blow the context window, so cap
    it: top eligible with reasons/docs, top partial with missing fields,
    counts for the rest. The agent keeps the FULL result for its guards."""
    results = obs["results"]
    # best-targeted schemes first: most verified constraints, then rows
    # without unverified free-text conditions; partial = closest to a
    # verdict first (fewest missing fields, most specific rule)
    eligible = sorted(
        (r for r in results if r["status"] == "eligible"),
        key=lambda r: (-r["match_score"], bool(r["other_conditions"].strip())))
    partial = sorted(
        (r for r in results if r["status"] == "partial"),
        key=lambda r: (len(r["missing_fields"]), -r["match_score"]))
    nq = obs["suggested_next_question"]
    if nq:
        # blocking_schemes can be thousands of names — with a pinned 8k
        # context that alone overflows the window; send count + examples
        nq = {"field": nq["field"], "question": nq["question"],
              "blocking_count": len(nq["blocking_schemes"]),
              "blocking_examples": nq["blocking_schemes"][:3]}
    return {
        "counts": {"eligible": len(eligible), "partial": len(partial),
                   "not_eligible": len(results) - len(eligible) - len(partial)},
        "eligible": [
            {"scheme_id": r["scheme_id"], "scheme_name": r["scheme_name"],
             "match_score": r["match_score"],
             "reasons": r["reasons"], "other_conditions": r["other_conditions"][:200],
             "documents_required": r["documents_required"]}
            for r in eligible[:max_eligible]
        ],
        "partial_top": [
            {"scheme_id": r["scheme_id"], "scheme_name": r["scheme_name"],
             "missing_fields": r["missing_fields"]}
            for r in partial[:max_partial]
        ],
        "suggested_next_question": nq,
        "summary": {"eligible": [r["scheme_name"] for r in eligible[:max_eligible]],
                    "partial": [r["scheme_name"] for r in partial[:max_partial]]},
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
        "documents_required": d["documents_required"][:15],
        # official FAQs (from the portal's FAQ API) — quote these when the
        # user asks practical questions (amounts, timelines, how to apply)
        "faqs": [{"q": f["question"], "a": f["answer"][:300]}
                 for f in (d.get("faqs") or [])[:3]],
    }


def update_profile(profile_store, field=None, value=None, fields=None, **_):
    """Write facts into the session profile (agent memory).

    Accepts either the batched form {"fields": {...}} (preferred — one LLM
    step saves everything) or the legacy {"field", "value"} pair.
    """
    batch = dict(fields) if isinstance(fields, dict) else {}
    if field is not None:
        batch[field] = value
    if not batch:
        return {"error": "provide 'fields' (dict) or 'field' + 'value'"}
    unknown = [k for k in batch if k not in KNOWN_FIELDS]
    if unknown:
        return {"error": f"unknown fields {unknown}. Known: {KNOWN_FIELDS}"}
    profile_store.update(batch)
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

4. update_profile — save what the user told you. Save ALL facts from the
   user's message in ONE call:
   input: {"fields": {"age": 34, "occupation": "artist", "annual_income": 40000}}
   allowed fields: age, annual_income, gender, category, occupation, state,
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
        if not ({"field", "value"} <= set(action_input) or "fields" in action_input):
            return {"error": "update_profile needs 'fields' (dict) or "
                             f"'field' + 'value', got {list(action_input.keys())}"}
        return update_profile(profile_store, **action_input)
    return {"error": f"unknown tool '{action}'"}
