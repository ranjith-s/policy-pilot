"""
llm.py — LLM backends.

OllamaClient  : local model via the Ollama HTTP API.
GeminiClient  : Google Gemini via the REST API (key from GEMINI_API_KEY env).
MockLLM       : deterministic scripted backend so the whole agent loop can be
                developed and tested without any LLM service (CI, demos).

All expose .chat(messages) -> str
"""

import json
import os
import time
import urllib.error
import urllib.request


class OllamaClient:
    def __init__(self, model="qwen3:4b-instruct", host="http://localhost:11434", temperature=0.0):
        self.model = model
        self.host = host
        self.temperature = temperature

    def chat(self, messages):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",  # ask Ollama to constrain output to valid JSON
            "keep_alive": "30m",  # avoid paying model reload between turns
            "options": {"temperature": self.temperature},
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        # retry transient errors once (model loading / swap on small GPUs
        # can cause a slow first response or a spurious 500)
        for attempt in (1, 2):
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    data = json.loads(resp.read())
                return data["message"]["content"]
            except (urllib.error.HTTPError, TimeoutError, OSError):
                if attempt == 2:
                    raise
                time.sleep(3)


class GeminiClient:
    """Google Gemini backend (REST, stdlib only).

    Get a free API key at https://aistudio.google.com/apikey and set:
        GEMINI_API_KEY=...   (env var; never hardcode keys)
    """

    def __init__(self, model="gemini-2.5-flash", api_key=None, temperature=0.0):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.temperature = temperature
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set (env var or api_key argument)")

    def chat(self, messages):
        system, contents = None, []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                contents.append({
                    "role": "model" if m["role"] == "assistant" else "user",
                    "parts": [{"text": m["content"]}],
                })
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",  # constrain to JSON
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "x-goog-api-key": self.api_key},
        )
        for attempt in (1, 2):  # one retry on transient errors / rate limits
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read())
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (urllib.error.HTTPError, TimeoutError, OSError) as e:
                if attempt == 2 or (isinstance(e, urllib.error.HTTPError)
                                    and e.code in (400, 401, 403)):
                    raise
                time.sleep(3)


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
