"""
app.py — Streamlit chat UI for the Public Scheme Eligibility Assistant.

Run:  streamlit run app/app.py
Wraps the same Agent used by the CLI; verdicts still come only from the
deterministic rules engine.
"""

import csv
import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import Agent
from llm import OllamaClient, GeminiClient, OpenAIClient, MockLLM
import tools


@st.cache_resource
def warm_caches():
    """Preload the 38 MB corpus + semantic index once per server process so
    the first search of a session doesn't stall."""
    tools._corpus()
    tools._get_semantic_index()
    return True

st.set_page_config(page_title="Scheme Eligibility Assistant", page_icon="🏛️",
                   layout="wide", initial_sidebar_state="expanded")
warm_caches()

# ------------------------------------------------------------------ style --
st.markdown("""
<style>
/* hero banner */
.hero {
  background: linear-gradient(120deg, #12356b 0%, #1a4fa0 55%, #2d6bd0 100%);
  border-radius: 18px; padding: 1.6rem 2rem 1.4rem;
  color: #fff; margin-bottom: 0.4rem;
  box-shadow: 0 8px 24px rgba(18, 53, 107, .18);
}
.hero h1 { color:#fff; font-size: 1.65rem; margin: 0 0 .3rem; }
.hero p  { color:#dbe6f7; margin: 0; font-size: .95rem; }
.hero .flag { display:inline-block; width: 46px; height: 5px; border-radius: 3px;
  background: linear-gradient(90deg,#ff9933 33%,#fff 33% 66%,#138808 66%);
  margin-bottom: .6rem; }

/* stat chips */
.stat { background:#fff; border:1px solid #e3eaf5; border-radius:14px;
  padding:.7rem 1rem; text-align:center;
  box-shadow: 0 2px 8px rgba(26,79,160,.06); }
.stat b { font-size:1.25rem; color:#1a4fa0; display:block; }
.stat span { font-size:.78rem; color:#5b6b85; }

/* chat bubbles */
[data-testid="stChatMessage"] { border-radius: 16px; padding: .9rem 1.1rem;
  margin-bottom: .35rem; box-shadow: 0 2px 10px rgba(18,53,107,.05); }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  background: #e8f0fe; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
  background: #ffffff; border: 1px solid #e3eaf5; }

/* example prompt buttons */
div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
  border-radius: 999px; border: 1px solid #c9d8f0; background:#fff;
  color:#1a4fa0; font-size:.85rem; }
div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
  border-color:#1a4fa0; background:#f0f5ff; }

/* sidebar */
section[data-testid="stSidebar"] { background:#0f2b57; }
section[data-testid="stSidebar"] * { color:#e8eefc !important; }
section[data-testid="stSidebar"] input {
  color:#1a2233 !important; background:#fff !important; }
section[data-testid="stSidebar"] hr { border-color:#2a4d8f; }

.disclaimer { background:#fff8ec; border:1px solid #f0ddb8; color:#7a5a17;
  border-radius:12px; padding:.65rem .9rem; font-size:.82rem; margin-top:.6rem; }
</style>
""", unsafe_allow_html=True)

STEP_LABELS = {
    "search_schemes": "🔍 searching schemes",
    "run_eligibility_check": "⚖️ checking eligibility rules",
    "get_scheme_details": "📄 fetching scheme details",
    "update_profile": "📝 noting your details",
    "ask_user": "❓ preparing a question",
    "final_answer": "✅ writing the answer",
}

EXAMPLES = [
    "I'm a 62 year old artist, what support can I get?",
    "I'm a widow living in Delhi",
    "I want to start a small business as an SC woman entrepreneur",
    "I'm a farmer in Tamil Nadu looking for support",
]


@st.cache_data
def coverage_stats():
    n_rules = 0
    with open(ROOT / "data" / "scheme_rules.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("rules_source") or "").strip():
                n_rules += 1
    with open(ROOT / "data" / "rag_corpus.json", encoding="utf-8") as f:
        n_total = len(json.load(f))
    return n_total, n_rules


# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.title("🏛️ Scheme Assistant")
    backend = st.radio("LLM backend",
                       ["Local Ollama", "Google Gemini", "OpenAI (ChatGPT)",
                        "Mock (no LLM)"],
                       help="Mock = scripted agent for demos without any model")
    if backend == "Local Ollama":
        model = st.text_input("Ollama model",
                              os.environ.get("OLLAMA_MODEL", "qwen3:4b-instruct"))
        host = st.text_input("Ollama host", "http://localhost:11434")
        api_key = ""
    elif backend == "Google Gemini":
        model = st.text_input("Gemini model", "gemini-2.5-flash")
        api_key = st.text_input("Gemini API key", os.environ.get("GEMINI_API_KEY", ""),
                                type="password",
                                help="Free key: https://aistudio.google.com/apikey — "
                                     "or set the GEMINI_API_KEY env var")
        host = ""
    elif backend == "OpenAI (ChatGPT)":
        model = st.text_input("OpenAI model", "gpt-4o-mini")
        api_key = st.text_input("OpenAI API key", os.environ.get("OPENAI_API_KEY", ""),
                                type="password",
                                help="Key: https://platform.openai.com/api-keys — "
                                     "or set the OPENAI_API_KEY env var")
        host = ""
    else:
        model = host = api_key = ""

    if st.button("🔄 New session", use_container_width=True):
        for k in ("agent", "chat", "pending_question", "last_steps"):
            st.session_state.pop(k, None)
        st.rerun()

    if "agent" in st.session_state and st.session_state.agent.profile:
        st.divider()
        st.subheader("🧾 Your profile")
        for k, v in st.session_state.agent.profile.items():
            st.markdown(f"- **{k.replace('_', ' ')}**: {v}")

    if st.session_state.get("last_steps"):
        st.divider()
        st.subheader("🤖 Agent steps (last turn)")
        for i, s in enumerate(st.session_state.last_steps, 1):
            with st.expander(f"{i}. {STEP_LABELS.get(s['action'], s['action'])}"):
                st.json({"thought": s.get("thought", ""),
                         "action_input": s.get("action_input", {})})


# ------------------------------------------------------------------- hero --
n_total, n_rules = coverage_stats()
st.markdown(f"""
<div class="hero">
  <div class="flag"></div>
  <h1>Public Scheme Eligibility Assistant</h1>
  <p>Tell me about yourself — I'll search {n_total:,} Indian government schemes,
  check the eligibility rules, and tell you what to do next.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, num, label in (
    (c1, f"{n_total:,}", "schemes in knowledge base"),
    (c2, f"{n_rules:,}", "with rule-checked eligibility"),
    (c3, "36", "states & UTs covered"),
    (c4, "0", "verdicts guessed by the LLM"),
):
    col.markdown(f'<div class="stat"><b>{num}</b><span>{label}</span></div>',
                 unsafe_allow_html=True)

st.markdown('<div class="disclaimer">⚠️ Results are <b>indicative only</b> — always '
            'verify on the <a href="https://www.myscheme.gov.in">official myScheme '
            'portal</a> before applying.</div>', unsafe_allow_html=True)
st.write("")

# ------------------------------------------------------------------ agent --
def make_agent():
    if backend == "Mock (no LLM)":
        llm = MockLLM()
    elif backend == "Google Gemini":
        llm = GeminiClient(model=model, api_key=api_key)
    elif backend == "OpenAI (ChatGPT)":
        llm = OpenAIClient(model=model, api_key=api_key)
    else:
        llm = OllamaClient(model=model, host=host)
    return Agent(llm)


if backend in ("Google Gemini", "OpenAI (ChatGPT)") and not api_key:
    st.info("Enter your API key in the sidebar to start.")
    st.stop()

config = (backend, model, host, api_key)
if "agent" not in st.session_state or st.session_state.get("config") != config:
    st.session_state.agent = make_agent()
    st.session_state.config = config
    st.session_state.chat = []
    st.session_state.pending_question = False
    st.session_state.last_steps = []

# ------------------------------------------------------------------- chat --
if not st.session_state.chat:
    st.caption("✨ Try one of these:")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, key=f"ex_{ex[:20]}", type="secondary",
                      use_container_width=True):
            st.session_state.queued_prompt = ex

for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)

prompt = st.chat_input("Describe yourself or answer the agent's question…")
if not prompt:
    prompt = st.session_state.pop("queued_prompt", None)

if prompt:
    st.session_state.chat.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    agent = st.session_state.agent
    with st.chat_message("assistant"):
        status = st.status("🤔 thinking…", expanded=True)
        agent.on_step = lambda act: status.write(
            STEP_LABELS.get(act["action"], act["action"])
            + (f" · {act['elapsed']}s" if act.get("elapsed") else ""))
        try:
            if st.session_state.pending_question:
                result = agent.answer_question(prompt)
            else:
                result = agent.run_turn(prompt)
        except Exception as e:
            status.update(label="backend error", state="error")
            st.error(f"LLM backend failed ({e}). Is Ollama running and the "
                     f"model pulled? Or switch to **Mock (no LLM)** in the sidebar.")
            st.stop()
        status.update(label="done", state="complete", expanded=False)

        st.session_state.pending_question = result["type"] == "question"
        st.session_state.last_steps = result["steps"]
        prefix = "**The agent asks:** " if st.session_state.pending_question else ""
        st.markdown(prefix + result["text"])
    st.session_state.chat.append(("assistant", prefix + result["text"]))
    st.rerun()
