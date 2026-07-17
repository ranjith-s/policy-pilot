"""
server.py — FastAPI backend for the React UI (PolicyPilot).

Thin HTTP wrapper around FunnelAgent: sessions in memory, structured JSON
responses (the same `data` payload the funnel builds for every turn).

Run:  python -m uvicorn api.server:app --port 8000 --reload
"""

import sys
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from funnel import FunnelAgent               # noqa: E402
from llm import OllamaClient, MockLLM        # noqa: E402
import tools                                  # noqa: E402
from engine import _load_rules                # noqa: E402

app = FastAPI(title="PolicyPilot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: dict[str, FunnelAgent] = {}
MAX_SESSIONS = 200


def _make_agent(backend: str = "ollama", model: str | None = None) -> FunnelAgent:
    if backend == "mock":
        return FunnelAgent(MockLLM())
    return FunnelAgent(OllamaClient(model=model or "qwen3:4b-instruct"))


class ChatIn(BaseModel):
    message: str = ""
    session_id: str | None = None
    backend: str = "ollama"          # "ollama" | "mock"
    model: str | None = None
    # structured answer from a UI control (dropdown/chips) — when both are
    # set, the turn skips the LLM entirely
    field: str | None = None
    value: str | int | bool | None = None


@app.on_event("startup")
def warm():
    """Preload corpus + rules so the first turn doesn't pay the load."""
    tools._corpus()
    _load_rules()


@app.get("/api/stats")
def stats():
    corpus = tools._corpus()
    return {"schemes": len(corpus), "rule_checked": len(tools._rules_ids())}


@app.post("/api/chat")
def chat(inp: ChatIn):
    sid = inp.session_id
    if not sid or sid not in SESSIONS:
        if len(SESSIONS) >= MAX_SESSIONS:          # crude LRU: drop oldest
            SESSIONS.pop(next(iter(SESSIONS)))
        sid = sid or uuid.uuid4().hex
        SESSIONS[sid] = _make_agent(inp.backend, inp.model)
    agent = SESSIONS[sid]

    if inp.field is not None and inp.value is not None:
        result = agent.answer_field(inp.field, inp.value)
    else:
        result = agent.run_turn(inp.message)
    return {
        "session_id": sid,
        "text": result["text"],
        "data": result.get("data"),
        "steps": [{"action": s["action"], "elapsed": s.get("elapsed")}
                  for s in result["steps"]],
    }


@app.post("/api/reset")
def reset(inp: ChatIn):
    if inp.session_id and inp.session_id in SESSIONS:
        del SESSIONS[inp.session_id]
    return {"ok": True}
