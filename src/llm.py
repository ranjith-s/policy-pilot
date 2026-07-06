"""
llm.py — LLM backends.

OllamaClient  : real local llama3.1 via the Ollama HTTP API.
MockLLM       : deterministic scripted backend so the whole agent loop can be
                developed and tested without Ollama running (CI, this sandbox).

Both expose .chat(messages) -> str
"""

import json
import urllib.request


class OllamaClient:
    def __init__(self, model="llama3.1", host="http://localhost:11434", temperature=0.1):
        self.model = model
        self.host = host
        self.temperature = temperature

    def chat(self, messages):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",  # ask Ollama to constrain output to valid JSON
            "options": {"temperature": self.temperature},
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return data["message"]["content"]


class MockLLM:
    """Rule-based stand-in that emits sensible agent actions.

    It reads the latest observation/user text in the transcript and picks the
    next action the way a well-behaved agent would. Good enough to exercise
    every code path (search -> update -> check -> ask -> details -> answer).
    """

    def chat(self, messages):
        transcript = json.dumps(messages).lower()
        last = messages[-1]["content"].lower()

        def act(thought, action, action_input):
            return json.dumps(
                {"thought": thought, "action": action, "action_input": action_input}
            )

        # 1. If the engine just ran, either ask the suggested question or finish.
        if '"suggested_next_question"' in last:
            obs = json.loads(messages[-1]["content"].split("observation:", 1)[-1].strip())
            nq = obs.get("suggested_next_question")
            eligible = obs.get("summary", {}).get("eligible", [])
            if nq and transcript.count('"ask_user"') < 2:
                return act(
                    f"Missing field '{nq['field']}' blocks {len(nq['blocking_schemes'])} schemes.",
                    "ask_user",
                    {"question": nq["question"]},
                )
            if eligible:
                return act(
                    "Engine returned verdicts; summarising for the user.",
                    "final_answer",
                    {"answer": "Based on the rules check you appear ELIGIBLE for: "
                               + ", ".join(eligible)
                               + ". Verify on the official portal before applying."},
                )
            return act(
                "No eligible schemes after check.",
                "final_answer",
                {"answer": "No exact matches found with the given details. "
                           "You can visit your nearest Common Service Centre for help."},
            )

        # 2. If the user just answered a question, save it then re-check.
        if "user answered:" in last:
            answer = messages[-1]["content"].split("user answered:", 1)[-1].strip()
            field = "annual_income" if any(c.isdigit() for c in answer) else "category"
            value = answer
            return act(
                f"Saving user's answer to profile field '{field}'.",
                "update_profile",
                {"field": field, "value": value},
            )

        # 3. After any update_profile, run the engine.
        if '"ok": true' in last and '"profile"' in last:
            return act("Profile updated; re-running eligibility.", "run_eligibility_check", {})

        # 4. After a search, extract facts from the original user message.
        if '"matches"' in last:
            return act("Found candidates; now recording what the user told us.",
                       "update_profile", self._first_fact(messages))

        # 5. Fresh user message: start with a search.
        return act("New request; searching schemes matching the user's words.",
                   "search_schemes", {"query": self._user_text(messages)})

    # helpers -------------------------------------------------------------
    def _user_text(self, messages):
        for m in reversed(messages):
            if m["role"] == "user" and "observation:" not in m["content"] \
                    and "user answered:" not in m["content"]:
                return m["content"][:80]
        return ""

    def _first_fact(self, messages):
        text = self._user_text(messages).lower()
        if "widow" in text:
            return {"field": "marital_status", "value": "widow"}
        if "artist" in text:
            return {"field": "occupation", "value": "artist"}
        if "entrepreneur" in text or "business" in text:
            return {"field": "occupation", "value": "entrepreneur"}
        if "faculty" in text or "professor" in text:
            return {"field": "occupation", "value": "faculty"}
        for word in text.split():
            if word.isdigit():
                return {"field": "age", "value": int(word)}
        return {"field": "occupation", "value": "unknown"}
