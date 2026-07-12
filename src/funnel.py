"""
funnel.py — Guided-funnel assistant: the conversation POLICY lives in code;
the LLM's only job is extracting facts from free text. It never picks tools
and never writes eligibility text, so answers can't stall, ramble, or fight
the guards.

One user turn = exactly ONE LLM call (fact extraction), then:
    engine over all rules (~40 ms)
 -> relevance-blended ranking (ms; semantic if Ollama is up, else keyword)
 -> deterministic answer: confirmed-eligible + strong candidates + the single
    highest-information follow-up question
'more' pages deeper into the ranked list with ZERO LLM calls.

Contrast with agent.py (free ReAct agent, kept for comparison): there the LLM
decides the flow each step — flexible, but a small local model may skip
follow-up questions or spend minutes writing long answers. Here the funnel is
guaranteed: wide first, narrowed every turn, always interactive.
"""

import json
import re
import time
from pathlib import Path

from engine import check_eligibility, get_next_question, KNOWN_FIELDS

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
PAGE = 5
# relevance (is this scheme about YOU) vs rule specificity (how precisely
# its rule targeted you) — relevance dominates so a farmer sees farm schemes
W_RELEVANCE, W_SPECIFICITY = 0.65, 0.35

DISCLAIMER = ("Results are indicative only — verify on the official myScheme "
              "portal (https://www.myscheme.gov.in) before applying.")

EXTRACT_SYSTEM = """You extract structured facts from a user's message for an Indian welfare-scheme eligibility profile.

Respond with ONLY one JSON object, no other text:
{"fields": {<only facts the user stated>}, "keywords": "<topic words for scheme search>"}

Allowed keys in "fields":
  age (number), annual_income (number, rupees per year), gender (male|female),
  category (SC|ST|OBC|EWS|General|BPL), occupation (lowercase, e.g. farmer, student, artist),
  state (Indian state name), marital_status (single|married|widow),
  has_bank_account (yes|no), land_owner (yes|no)

Rules:
- Include ONLY what the message states or clearly implies. Never guess.
- Convert amounts: "1.2 lakh" -> 120000, "40k" -> 40000.
- If the message answers the assistant's question shown in the context, map it to that field.
- "keywords" = the user's own topic/need words (e.g. "farmer irrigation loan"); "" if none.
"""

MORE_RE = re.compile(r"^\s*(more|show more|next|more schemes|show me more)\s*[.!]*\s*$", re.I)


class FunnelAgent:
    """Same public surface as agent.Agent: run_turn / answer_question /
    profile / on_step, so the CLI and Streamlit UI can swap them freely."""

    def __init__(self, llm, session_id=None, on_step=None, use_semantic=True):
        self.llm = llm
        self.on_step = on_step
        self.profile = {}
        self.query_text = ""            # accumulated topic words across turns
        self.last_question = None       # so extraction can interpret "Kerala"
        self.eligible = []              # full ranked lists (session state)
        self.candidates = []
        self.shown = {"eligible": 0, "candidates": 0}   # pagination cursors
        self.use_semantic = use_semantic
        self._sem = None
        self._sem_tried = False
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        LOGS_DIR.mkdir(exist_ok=True)
        self.trace_path = LOGS_DIR / "agent_trace.jsonl"

    # ------------------------------------------------------------- helpers --
    def _trace(self, **event):
        event.update(session=self.session_id, mode="funnel", ts=round(time.time(), 2))
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _step(self, steps, action, action_input, thought="", elapsed=None):
        act = {"thought": thought, "action": action, "action_input": action_input}
        if elapsed is not None:
            act["elapsed"] = round(elapsed, 1)
        self._trace(event="agent_step", step=len(steps) + 1, **act)
        steps.append(act)
        if self.on_step:
            self.on_step(act)

    def _semantic_index(self):
        if self.use_semantic and not self._sem_tried:
            self._sem_tried = True
            try:
                from embeddings import SemanticIndex
                if SemanticIndex.available():
                    self._sem = SemanticIndex()
            except Exception:
                self._sem = None
        return self._sem

    # ------------------------------------------------------ fact extraction --
    def _parse_json(self, raw):
        text = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.M).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                raise
            return json.loads(m.group(0))

    def _extract(self, user_message):
        """ONE LLM call: message -> {fields, keywords}. Failure-tolerant:
        one repair attempt, then proceed with nothing extracted."""
        context = ""
        if self.profile:
            context += f"Profile so far: {json.dumps(self.profile)}. "
        if self.last_question:
            context += f'The assistant just asked: "{self.last_question}". '
        messages = [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": context + f'User message: "{user_message}"'},
        ]
        for attempt in (1, 2):
            try:
                obj = self._parse_json(self.llm.chat(messages))
                fields = obj.get("fields") or {}
                keywords = str(obj.get("keywords") or "")
                break
            except Exception:
                if attempt == 2:
                    self._trace(event="extract_failure", message=user_message[:200])
                    return {}, ""
                messages.append({"role": "user",
                                 "content": "Not valid JSON. Respond again with ONLY "
                                            'the JSON object {"fields": {...}, "keywords": "..."}.'})
        clean = {}
        for k, v in (fields or {}).items():
            if k in KNOWN_FIELDS and str(v).strip().lower() not in (
                    "", "none", "null", "unknown", "n/a", "not stated"):
                clean[k] = v
        return clean, keywords

    # ------------------------------------------------------------- ranking --
    def _relevance(self, ids):
        """0..1 relevance per scheme id from the user's own words.
        Semantic when the index + Ollama are up; keyword otherwise; all-0
        when the user gave no topic words (ranking falls back to specificity)."""
        text = self.query_text.strip()
        if not text:
            return {}
        sem = self._semantic_index()
        scores = None
        if sem is not None:
            try:
                scores = sem.all_scores(text)
            except Exception:
                scores = None
        if scores is None:                       # keyword fallback
            from tools import _corpus
            words = [w for w in text.lower().split() if len(w) > 2]
            scores = {}
            for sid in ids:
                doc = _corpus().get(sid)
                scores[sid] = float(sum(doc["search_text"].count(w) for w in words)) if doc else 0.0
        vals = [scores.get(sid, 0.0) for sid in ids]
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            return {sid: 0.0 for sid in ids}
        return {sid: (scores.get(sid, 0.0) - lo) / (hi - lo) for sid in ids}

    def _rank(self, results):
        ids = [r["scheme_id"] for r in results]
        rel = self._relevance(ids)
        max_spec = max((r["match_score"] for r in results), default=1) or 1
        for r in results:
            r["_score"] = (W_RELEVANCE * rel.get(r["scheme_id"], 0.0)
                           + W_SPECIFICITY * r["match_score"] / max_spec)
        self.eligible = sorted((r for r in results if r["status"] == "eligible"),
                               key=lambda r: -r["_score"])
        self.candidates = sorted((r for r in results if r["status"] == "partial"),
                                 key=lambda r: (-r["_score"], len(r["missing_fields"])))
        self.shown = {"eligible": 0, "candidates": 0}

    # ------------------------------------------------------------ rendering --
    @staticmethod
    def _human(field):
        return field.replace("_", " ")

    def _render_page(self, header=True):
        out = []
        e_from, c_from = self.shown["eligible"], self.shown["candidates"]
        e_batch = self.eligible[e_from:e_from + PAGE]
        c_batch = self.candidates[c_from:c_from + PAGE]
        self.shown["eligible"] = e_from + len(e_batch)
        self.shown["candidates"] = c_from + len(c_batch)

        if e_batch:
            out.append(f"ELIGIBLE by the rules check — {e_from + 1}-{e_from + len(e_batch)} "
                       f"of {len(self.eligible)}:")
            for i, r in enumerate(e_batch, e_from + 1):
                out.append(f"  {i}. {r['scheme_name']}")
                if r["reasons"]:
                    out.append(f"     why: {'; '.join(r['reasons'][:4])}")
                if r["documents_required"]:
                    out.append(f"     documents: {', '.join(r['documents_required'][:4])}")
        elif header and self.eligible:
            out.append("No further eligible schemes.")

        if c_batch:
            out.append(f"LIKELY MATCHES (answer below to confirm) — {c_from + 1}-"
                       f"{c_from + len(c_batch)} of {len(self.candidates)}:")
            for i, r in enumerate(c_batch, c_from + 1):
                need = ", ".join(self._human(f) for f in r["missing_fields"][:4])
                out.append(f"  {i}. {r['scheme_name']}  (need: {need})")

        if not out:
            out.append("No more results — that's everything the rules check found.")
        return out

    def _compose(self, next_q):
        lines = self._render_page()
        if not self.eligible and not self.candidates:
            lines = ["No schemes matched the rules check yet. Try describing your "
                     "situation differently, or visit your nearest Common Service Centre."]
        if next_q:
            lines.append(f"To narrow this down: {next_q['question']}")
            self.last_question = next_q["question"]
        else:
            self.last_question = None
        remaining = (len(self.eligible) - self.shown["eligible"]
                     + len(self.candidates) - self.shown["candidates"])
        if remaining > 0:
            lines.append(f"(type 'more' for the next {min(PAGE, remaining)} of "
                         f"{remaining} remaining schemes)")
        lines.append(DISCLAIMER)
        return "\n".join(lines)

    # ------------------------------------------------------------ main turn --
    def run_turn(self, user_message):
        self._trace(event="user_message", text=user_message)
        steps = []

        # 'more' = pure pagination: zero LLM calls, instant
        if MORE_RE.match(user_message):
            if not (self.eligible or self.candidates):
                return {"type": "answer", "steps": steps,
                        "text": "Tell me about yourself first — e.g. \"I'm a farmer "
                                "in Tamil Nadu\" — and I'll list matching schemes."}
            self._step(steps, "show_more", {"cursor": dict(self.shown)},
                       "Paging through the ranked list (no LLM needed).")
            lines = self._render_page(header=False)
            if self.last_question:
                lines.append(f"To narrow this down: {self.last_question}")
            lines.append(DISCLAIMER)
            text = "\n".join(lines)
            self._trace(event="final_answer", text=text[:500])
            return {"type": "answer", "text": text, "steps": steps}

        # 1. ONE LLM call: extract facts + topic words
        t0 = time.time()
        fields, keywords = self._extract(user_message)
        if fields:
            self.profile.update(fields)
        if keywords:
            merged = f"{self.query_text} {keywords}".split()
            self.query_text = " ".join(dict.fromkeys(merged))[:300]
        self._step(steps, "update_profile", {"fields": fields, "keywords": keywords},
                   "Extracted the facts stated in the message.", time.time() - t0)

        # 2. deterministic engine over every rule-annotated scheme
        t0 = time.time()
        results = check_eligibility(self.profile)
        self._step(steps, "run_eligibility_check",
                   {"profile": dict(self.profile)},
                   "Engine checked every rule-annotated scheme.", time.time() - t0)

        # 3. relevance-blended ranking + highest-information next question,
        #    chosen from the TOP candidates so it unblocks what the user
        #    actually cares about
        t0 = time.time()
        self._rank(results)
        next_q = get_next_question(self.profile, self.candidates[:50])
        self._step(steps, "rank_schemes",
                   {"eligible": len(self.eligible), "candidates": len(self.candidates),
                    "query": self.query_text},
                   "Ranked by relevance to your words + rule specificity.",
                   time.time() - t0)

        text = self._compose(next_q)
        self._trace(event="final_answer", text=text[:500])
        return {"type": "answer", "text": text, "steps": steps}

    def answer_question(self, user_answer):
        return self.run_turn(user_answer)
