"""
main.py — CLI chat interface for the Public Scheme Eligibility Assistant.

Usage:
    python src/main.py                 # real agent, local Ollama llama3.1
    python src/main.py --mock          # scripted MockLLM (no Ollama needed)
    python src/main.py --show-trace    # print each agent step as it happens
    python src/main.py --model llama3.1:8b --host http://localhost:11434
"""

import argparse
import sys
from pathlib import Path

# Windows consoles default to cp1252, which can't print ₹ etc.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import Agent
from llm import OllamaClient, GeminiClient, OpenAIClient, MockLLM

BANNER = """
================================================================
  Public Scheme Eligibility Assistant  (agentic CLI prototype)
================================================================
Tell me about yourself and what you're looking for, e.g.:
  "I'm a 62 year old artist, what support can I get?"
  "I'm a widow living in Delhi"
Type 'profile' to see stored facts, 'reset' to start over, 'quit' to exit.

NOTE: Results are indicative only. Always verify on the official
myScheme portal (https://www.myscheme.gov.in) before applying.
================================================================
"""


STEP_LABELS = {
    "search_schemes": "searching schemes",
    "run_eligibility_check": "checking eligibility rules",
    "get_scheme_details": "fetching scheme details",
    "update_profile": "noting your details",
    "ask_user": "preparing a question",
    "final_answer": "writing the answer",
}


def make_step_printer(show_trace):
    """Live progress line per agent step, so the CLI never looks frozen."""
    def on_step(act):
        label = STEP_LABELS.get(act["action"], act["action"])
        took = f" ({act['elapsed']}s)" if act.get("elapsed") else ""
        print(f"  … {label}{took}", flush=True)
        if show_trace:
            print(f"      {act['action']}({act['action_input']})", flush=True)
            if act.get("thought"):
                print(f"      thought: {act['thought']}", flush=True)
    return on_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="use scripted MockLLM (no Ollama)")
    ap.add_argument("--gemini", action="store_true",
                    help="use Google Gemini (needs GEMINI_API_KEY env var)")
    ap.add_argument("--chatgpt", "--openai", action="store_true", dest="chatgpt",
                    help="use OpenAI/ChatGPT (needs OPENAI_API_KEY env var)")
    ap.add_argument("--model", default=None,
                    help="default: qwen3:4b-instruct (Ollama) / gemini-2.5-flash "
                         "(--gemini) / gpt-4o-mini (--chatgpt)")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--show-trace", action="store_true", help="print agent steps live")
    args = ap.parse_args()

    if args.mock:
        llm = MockLLM()
    elif args.gemini:
        llm = GeminiClient(model=args.model or "gemini-2.5-flash")
    elif args.chatgpt:
        llm = OpenAIClient(model=args.model or "gpt-4o-mini")
    else:
        llm = OllamaClient(model=args.model or "qwen3:4b-instruct", host=args.host)
    on_step = make_step_printer(args.show_trace)
    agent = Agent(llm, on_step=on_step)

    print(BANNER)
    if args.mock:
        print("  (running with MockLLM — deterministic scripted agent)\n")

    pending_question = False
    while True:
        try:
            user = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break
        if not user:
            continue
        if user.lower() in ("quit", "exit"):
            print("bye!")
            break
        if user.lower() == "profile":
            print(f"stored profile: {agent.profile}")
            continue
        if user.lower() == "reset":
            agent = Agent(llm, on_step=on_step)
            pending_question = False
            print("(new session)")
            continue

        try:
            result = agent.answer_question(user) if pending_question else agent.run_turn(user)
        except Exception as e:
            print(f"\n[error] LLM backend failed ({e}). Is Ollama running "
                  f"and the model pulled? Try again, or use --mock.\n")
            continue
        pending_question = result["type"] == "question"

        prefix = "agent asks" if pending_question else "agent"
        print(f"\n{prefix} > {result['text']}\n")


if __name__ == "__main__":
    main()
