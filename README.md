# Public Scheme Eligibility Assistant (Agentic CLI Prototype)

An agentic AI assistant that helps citizens discover Indian government welfare
schemes and check their eligibility — with all eligibility verdicts coming from
a **deterministic rules engine**, never from the LLM.

## Architecture

```
User (CLI)
   │
   ▼
ReAct Agent Loop (src/agent.py)          ← LLM decides the next action each step
   │  {"thought", "action", "action_input"}
   ▼
Toolbox (src/tools.py)
   ├── search_schemes          keyword+metadata search over rag_corpus.json
   ├── run_eligibility_check   → engine.py (deterministic, source of truth)
   ├── get_scheme_details      benefits / application steps lookup
   ├── update_profile          session memory (facts the user shared)
   ├── ask_user                human-in-the-loop question (ends the turn)
   └── final_answer            ends the turn with an answer
```

### Guardrails (enforced in code, not just prompt)
- **Verdict guard** — a `final_answer` claiming eligibility is rejected unless
  the engine's latest run actually returned that scheme as eligible.
- **Loop cap** — max 6 tool calls per user turn.
- **Question cap** — max 2 `ask_user` rounds per conversation, then best-effort answer.
- **JSON repair** — one retry on malformed LLM output, then a non-LLM fallback
  answer built from the engine's last results (a demo can never dead-end).
- **Trace log** — every thought/action/observation appended to `logs/agent_trace.jsonl`.

## Run

```bash
# with local Ollama (needs `ollama pull llama3.1` and ollama serving)
python src/main.py --show-trace

# without Ollama (scripted MockLLM, exercises the full loop deterministically)
python src/main.py --mock --show-trace
```

Try: `I'm a 65 year old artist, what support can I get?`
CLI commands: `profile` (show stored facts), `reset`, `quit`.

## Tests

```bash
python tests/test_engine.py     # 7 personas, 19 assertions against scheme_rules.csv
```

## Data

- `data/rag_corpus.json` — cleaned scheme documents (from `prepare_scheme_data.py`,
  parsed from myScheme portal scrape). Currently 5 sample schemes; the pipeline
  scales to the full ~4000-scheme scrape unchanged.
- `data/scheme_rules.csv` — manually annotated eligibility rules (Tier 2).
  Blank cell = criterion not applicable. `female_any_category` in the category
  column means women of any category qualify (e.g. Stand-Up India).

## Extending

- **Add a scheme**: add its doc to the corpus (rerun the prep script) + one
  annotated row in `scheme_rules.csv`. No code changes.
- **Swap LLM**: implement `.chat(messages) -> str` in `src/llm.py` (e.g. a cloud
  API client) and pass it to `Agent`.
- **UI**: `Agent.run_turn()` / `answer_question()` return
  `{"type": "answer"|"question", "text", "steps"}` — a Streamlit chat wrapper
  maps onto this directly.

## Limitations / responsible use

Results are indicative only — always verify on https://www.myscheme.gov.in
before applying. Only 5 schemes annotated so far; documents-required and FAQ
data not yet scraped; English only; local 8B model may occasionally produce
malformed actions (handled by repair + fallback).
