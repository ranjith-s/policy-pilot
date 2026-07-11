"""
app.py — Streamlit chat UI for the Public Scheme Eligibility Assistant.

Run:  streamlit run app/app.py
Wraps the same Agent used by the CLI; verdicts still come only from the
deterministic rules engine.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import Agent
from llm import OllamaClient, MockLLM

st.set_page_config(page_title="Scheme Eligibility Assistant", page_icon="🏛️",
                   layout="wide")

STEP_LABELS = {
    "search_schemes": "🔍 searching schemes",
    "run_eligibility_check": "⚖️ checking eligibility rules",
    "get_scheme_details": "📄 fetching scheme details",
    "update_profile": "📝 noting your details",
    "ask_user": "❓ preparing a question",
    "final_answer": "✅ writing the answer",
}


# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.title("🏛️ Scheme Assistant")
    use_mock = st.toggle("Mock LLM (no Ollama needed)", value=False,
                         help="Scripted agent for demos without a local model")
    model = st.text_input("Ollama model", "qwen3:4b-instruct", disabled=use_mock)
    host = st.text_input("Ollama host", "http://localhost:11434", disabled=use_mock)

    if st.button("🔄 New session"):
        for k in ("agent", "chat", "pending_question", "last_steps"):
            st.session_state.pop(k, None)
        st.rerun()

    if "agent" in st.session_state and st.session_state.agent.profile:
        st.subheader("Your profile (session memory)")
        for k, v in st.session_state.agent.profile.items():
            st.write(f"- **{k}**: {v}")

    if st.session_state.get("last_steps"):
        st.subheader("Agent steps (last turn)")
        for i, s in enumerate(st.session_state.last_steps, 1):
            with st.expander(f"{i}. {STEP_LABELS.get(s['action'], s['action'])}"):
                st.json({"thought": s.get("thought", ""),
                         "action_input": s.get("action_input", {})})

    st.caption("Results are **indicative only** — always verify on the "
               "[official myScheme portal](https://www.myscheme.gov.in) "
               "before applying.")


# ------------------------------------------------------------------ agent --
def make_agent():
    llm = MockLLM() if use_mock else OllamaClient(model=model, host=host)
    return Agent(llm)


config = (use_mock, model, host)
if "agent" not in st.session_state or st.session_state.get("config") != config:
    st.session_state.agent = make_agent()
    st.session_state.config = config
    st.session_state.chat = []
    st.session_state.pending_question = False
    st.session_state.last_steps = []

# ------------------------------------------------------------------- chat --
st.markdown("#### Tell me about yourself and what support you're looking for")
st.caption('e.g. *"I\'m a 62 year old artist, what support can I get?"* — '
           'the agent asks follow-up questions when it needs more details.')

for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)

if prompt := st.chat_input("Describe yourself or answer the agent's question…"):
    st.session_state.chat.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    agent = st.session_state.agent
    with st.chat_message("assistant"):
        status = st.status("thinking…", expanded=True)
        agent.on_step = lambda act: status.write(
            STEP_LABELS.get(act["action"], act["action"]))
        try:
            if st.session_state.pending_question:
                result = agent.answer_question(prompt)
            else:
                result = agent.run_turn(prompt)
        except Exception as e:
            status.update(label="backend error", state="error")
            st.error(f"LLM backend failed ({e}). Is Ollama running and the "
                     f"model pulled? Or enable **Mock LLM** in the sidebar.")
            st.stop()
        status.update(label="done", state="complete", expanded=False)

        st.session_state.pending_question = result["type"] == "question"
        st.session_state.last_steps = result["steps"]
        prefix = "**The agent asks:** " if st.session_state.pending_question else ""
        st.markdown(prefix + result["text"])
    st.session_state.chat.append(("assistant", prefix + result["text"]))
    st.rerun()
