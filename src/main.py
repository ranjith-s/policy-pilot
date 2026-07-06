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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import Agent
from llm import OllamaClient, MockLLM

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


def print_steps(steps):
    for i, s in enumerate(steps, 1):
        print(f"    [{i}] {s['action']}({s['action_input']})")
        if s.get("thought"):
            print(f"        thought: {s['thought']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="use scripted MockLLM (no Ollama)")
    ap.add_argument("--model", default="llama3.1")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--show-trace", action="store_true", help="print agent steps live")
    args = ap.parse_args()

    llm = MockLLM() if args.mock else OllamaClient(model=args.model, host=args.host)
    agent = Agent(llm)

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
            agent = Agent(llm)
            pending_question = False
            print("(new session)")
            continue

        result = agent.answer_question(user) if pending_question else agent.run_turn(user)
        pending_question = result["type"] == "question"

        if args.show_trace:
            print("  --- agent steps ---")
            print_steps(result["steps"])
            print("  -------------------")

        prefix = "agent asks" if pending_question else "agent"
        print(f"\n{prefix} > {result['text']}\n")


if __name__ == "__main__":
    main()
