"""
agent.py — The ReAct agent loop.

LLM output protocol (one JSON object per step):
    {"thought": "...", "action": "<tool name>", "action_input": {...}}

Loop guarantees (guardrails enforced in CODE, not prompt):
  * max MAX_STEPS tool calls per user turn
  * max MAX_QUESTIONS ask_user rounds per conversation
  * verdict guard: a final_answer naming a scheme as eligible is rejected
    unless the engine's last run actually returned it as eligible
  * malformed JSON gets one repair attempt, then a safe fallback
  * every step appended to logs/agent_trace.jsonl
"""

import json
import re
import time
from pathlib import Path

from tools import TOOL_SCHEMAS, dispatch

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
MAX_STEPS = 8
MAX_QUESTIONS = 2

SYSTEM_PROMPT = f"""You are a Public Scheme Eligibility Assistant agent for Indian government welfare schemes.

You work step by step. At every step respond with ONLY one JSON object:
{{"thought": "<your reasoning>", "action": "<tool name>", "action_input": {{...}}}}

{TOOL_SCHEMAS}

STRICT RULES:
- NEVER state or guess eligibility yourself. Verdicts come ONLY from run_eligibility_check.
- When the user tells you facts, save EVERY one with update_profile BEFORE checking eligibility — including occupation (artist, farmer, entrepreneur, faculty...), age, income, gender, state, marital status.
- In search_schemes, only pass "state" if the user named their state; only pass "category" if clearly needed. When unsure, use query words alone.
- ALWAYS call run_eligibility_check once after saving the user's facts. It checks every rule-annotated scheme at once — even schemes search did not return — so run it even if search results look unpromising.
- If run_eligibility_check reports a suggested_next_question and you have questions left, use ask_user with it. One question at a time.
- In your final_answer: state eligible schemes with reasons, required documents, and how to apply. Always add: "This is indicative only — verify on the official myScheme portal before applying."
- If nothing matches, say so honestly and suggest the nearest Common Service Centre.
- Keep "thought" to ONE short sentence.
- Respond ONLY with the JSON object. No other text.
"""


class Agent:
    def __init__(self, llm, session_id=None, on_step=None):
        self.llm = llm
        self.on_step = on_step            # live progress callback: fn(act_dict)
        self.profile = {}                 # session memory
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.last_engine_result = None    # for the verdict guard
        self.questions_asked = 0
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        LOGS_DIR.mkdir(exist_ok=True)
        self.trace_path = LOGS_DIR / "agent_trace.jsonl"

    # ------------------------------------------------------------------ #
    def _trace(self, **event):
        event.update(session=self.session_id, ts=round(time.time(), 2))
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _parse_action(self, raw):
        """Parse the LLM's JSON action; tolerate markdown fences / stray text."""
        text = raw.strip()
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.M).strip()
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)  # grab the outermost {...}
            if not m:
                raise
            obj = json.loads(m.group(0))
        if not isinstance(obj, dict) or "action" not in obj:
            raise ValueError("no 'action' key")
        obj.setdefault("thought", "")
        obj.setdefault("action_input", {})
        # small local models sometimes emit action_input as a bare string;
        # coerce it into the dict shape each tool expects
        if not isinstance(obj["action_input"], dict):
            key = {"final_answer": "answer", "ask_user": "question",
                   "search_schemes": "query"}.get(obj["action"])
            obj["action_input"] = {key: obj["action_input"]} if key else {}
        return obj

    def _violates_verdict_guard(self, answer_text):
        """True if the answer claims eligibility for a scheme the engine
        did not mark eligible in its most recent run.

        Strict by design: in an answer that makes a positive eligibility
        claim, every named non-eligible scheme must carry a qualifier
        ("possibly eligible", "more info needed"...) on the same line —
        otherwise the answer is rejected. Errs on the safe side.
        """
        text = answer_text.lower()
        if "eligible" not in text:
            return False
        if not self.last_engine_result:
            return True  # claiming eligibility without ever running the engine
        if not re.search(r"(?<!not )(?<!not-)eligible", text):
            return False  # only negative mentions ("not eligible")
        qualifiers = ("not eligible", "possibly eligible", "may be eligible",
                      "might be eligible", "partial", "more info",
                      "more information", "cannot verify", "could not verify",
                      "couldn't verify", "unable to verify")
        for r in self.last_engine_result["results"]:
            name = r["scheme_name"].lower()
            if r["status"] == "eligible" or name not in text:
                continue
            for line in text.splitlines():
                if name in line and not any(q in line for q in qualifiers):
                    return True
        return False

    # ------------------------------------------------------------------ #
    def run_turn(self, user_message):
        """Process one user message. Returns
        {"type": "answer"|"question", "text": ..., "steps": [...]}"""
        self.messages.append({"role": "user", "content": user_message})
        self._trace(event="user_message", text=user_message)
        steps = []
        engine_ran_this_turn = False

        for step_no in range(1, MAX_STEPS + 1):
            raw = self.llm.chat(self.messages)

            try:
                act = self._parse_action(raw)
            except Exception:
                # one repair attempt
                self.messages.append({
                    "role": "user",
                    "content": "Your last reply was not valid JSON. Respond again "
                               "with ONLY the JSON action object.",
                })
                try:
                    act = self._parse_action(self.llm.chat(self.messages))
                except Exception:
                    self._trace(event="parse_failure", raw=raw[:300])
                    return self._fallback(steps)

            self.messages.append({"role": "assistant", "content": json.dumps(act)})
            self._trace(event="agent_step", step=step_no, **act)
            steps.append(act)
            if self.on_step:
                self.on_step(act)
            action, inp = act["action"], act["action_input"]

            # ---- terminal actions ----
            if action == "final_answer":
                answer = inp.get("answer", "")
                # engine-first guard: with profile facts on file, the engine
                # must have been consulted THIS turn (fresh data) before any
                # final answer
                if self.profile and not engine_ran_this_turn \
                        and step_no < MAX_STEPS:
                    self._trace(event="engine_first_guard_triggered")
                    self.messages.append({
                        "role": "user",
                        "content": "GUARD: answer rejected — run_eligibility_check was "
                                   "not called this turn. First save any new fact the "
                                   "user just gave with update_profile, then call "
                                   "run_eligibility_check, then answer from its verdicts.",
                    })
                    continue
                if self._violates_verdict_guard(answer):
                    self._trace(event="verdict_guard_triggered", answer=answer[:200])
                    self.messages.append({
                        "role": "user",
                        "content": "GUARD: your answer claims eligibility not confirmed by "
                                   "run_eligibility_check. Re-answer using only the engine's "
                                   "verdicts, or run the check first.",
                    })
                    continue
                self._trace(event="final_answer", text=answer[:500])
                return {"type": "answer", "text": answer, "steps": steps}

            if action == "ask_user":
                if self.questions_asked >= MAX_QUESTIONS:
                    self.messages.append({
                        "role": "user",
                        "content": "GUARD: question limit reached. Give a final_answer "
                                   "with the best information you have.",
                    })
                    continue
                self.questions_asked += 1
                q = inp.get("question", "Could you tell me more?")
                self._trace(event="ask_user", question=q)
                return {"type": "question", "text": q, "steps": steps}

            # ---- normal tools ----
            obs = dispatch(action, inp, self.profile)
            if action == "run_eligibility_check" and "results" in obs:
                self.last_engine_result = obs
                engine_ran_this_turn = True
            self._trace(event="observation", tool=action,
                        observation=json.dumps(obs)[:1000])
            self.messages.append({
                "role": "user",
                "content": "observation: " + json.dumps(obs, ensure_ascii=False),
            })

        return self._fallback(steps)

    def answer_question(self, user_answer):
        """Feed the user's reply to an ask_user question back into the loop."""
        return self.run_turn(f"user answered: {user_answer}")

    def _fallback(self, steps):
        """Non-LLM safety net so a demo can never dead-end."""
        if self.last_engine_result:
            s = self.last_engine_result["summary"]
            text = ("Here is what the rules check found so far — Eligible: "
                    + (", ".join(s["eligible"]) or "none")
                    + ". Possibly eligible (more info needed): "
                    + (", ".join(s["partial"]) or "none")
                    + ". This is indicative only — verify on the official myScheme portal.")
        else:
            text = ("I couldn't complete the check this time. Please try rephrasing, "
                    "or visit https://www.myscheme.gov.in directly.")
        self._trace(event="fallback", text=text[:300])
        return {"type": "answer", "text": text, "steps": steps}
