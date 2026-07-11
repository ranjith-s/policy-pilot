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
  the engine's latest run actually returned that scheme as eligible; any
  non-eligible scheme it names must carry a qualifier ("possibly eligible",
  "more info needed") on the same line.
- **Engine-first guard** — with profile facts on file, a `final_answer` is
  rejected unless `run_eligibility_check` was called in the current turn
  (stale verdicts from earlier turns don't count).
- **Loop cap** — max 8 tool calls per user turn.
- **Question cap** — max 2 `ask_user` rounds per conversation, then best-effort answer.
- **JSON repair** — one retry on malformed LLM output, then a non-LLM fallback
  answer built from the engine's last results (a demo can never dead-end).
- **Trace log** — every thought/action/observation appended to `logs/agent_trace.jsonl`.

## Run

```bash
# web UI (recommended for demos) — http://localhost:8501
streamlit run app/app.py

# CLI with local Ollama (needs `ollama pull qwen3:4b-instruct` and ollama serving)
python src/main.py --show-trace

# pick a different local model
python src/main.py --model qwen2.5:7b

# cloud LLMs (set the key env var first — never hardcode keys)
python src/main.py --gemini            # GEMINI_API_KEY  (free: aistudio.google.com/apikey)
python src/main.py --chatgpt           # OPENAI_API_KEY  (platform.openai.com/api-keys)
python src/main.py --gemini --model gemini-2.5-pro
python src/main.py --chatgpt --model gpt-4o

# without Ollama (scripted MockLLM, exercises the full loop deterministically;
# the UI has the same toggle in its sidebar)
python src/main.py --mock --show-trace
```

Try: `I'm a 65 year old artist, what support can I get?`
CLI commands: `profile` (show stored facts), `reset`, `quit`.

## Tests

```bash
python tests/test_engine.py     # 7 personas, 19 assertions against scheme_rules.csv
```

## Data

- `data/rag_corpus.json` — cleaned scheme documents (from
  `pipeline/prepare_scheme_data.py`, parsed from the myScheme portal scrape).
  Full corpus: **4,682 schemes**.
- `data/scheme_rules.csv` — structured eligibility rules. Each row carries a
  `rules_source`: `manual` (5 hand-curated gold rows) or `llm` (extracted by
  `src/extract_rules.py`, see below). Blank cell = criterion not applicable.
  `female_any_category` in the category column means women of any category
  qualify (e.g. Stand-Up India). Rows with no structured criteria are ignored
  by the engine (safety guard: they can never yield an "eligible" verdict —
  such schemes are describe-only).

## LLM rule extraction (how the engine "knows" schemes)

The LLM plays two separate roles in this project:

1. **Offline data pipeline** — `src/extract_rules.py` reads each scheme's
   *official* `eligibility_text` and fills the structured rule columns
   (age/income/gender/category/occupation/...). Conservative by construction:
   a field is only set when the text states it explicitly; everything else is
   demoted to free-text `other_conditions`. Output is validated in Python
   against controlled vocabularies and checkpointed (resumable).
2. **Online conversation agent** — never decides eligibility. Query-time
   verdicts come only from the deterministic engine, so they are reproducible,
   auditable, and instant on a laptop.

**Measured extraction quality**: 88% field-level agreement against the 5
hand-curated gold rows (qwen3.5:27b); the residual disagreements require
world knowledge that is absent from the official text (e.g. inferring a bank
account requirement for a bank-loan scheme). Run the audit yourself:
`python src/extract_rules.py review --sample 10` shows extracted fields next
to the source text.

## Data freshness (weekly refresh)

`python pipeline/refresh.py` re-syncs everything with the live portal:
rescrape (rate-limited, resumable) → rebuild corpus → **content-hash diff** →
LLM re-extracts *only new/changed schemes* → merge rules (manual rows always
win) → rebuild embeddings → engine tests must pass. A typical week changes a
handful of schemes, so a refresh costs minutes, not the full 6-hour backfill.
Weekly cadence is recommended; `data/refresh_report.json` records each run.


## Semantic retrieval (full-corpus scale)

At ~4000 schemes, keyword search misses vocabulary mismatches ("wedding" vs
"marriage"). The hybrid retriever fixes this:

```bash
ollama pull nomic-embed-text
python src/embeddings.py build      # one-off, re-run when corpus changes
```

`search_schemes` then runs: metadata hard-filter -> semantic cosine search
(numpy over a pre-computed ~12 MB matrix; no vector DB needed) -> blended with
keyword score (0.7/0.3). If the index is missing or Ollama is down it silently
falls back to keyword search — retrieval degrades, never dies.

### Rules coverage honesty
Only schemes annotated in `scheme_rules.csv` can receive engine verdicts.
Search results carry `rules_available: true/false`; the agent may only
*describe* uncovered schemes and link the official page — the verdict guard
prevents it from claiming eligibility for them.

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
before applying. Known limitations:

- **LLM-extracted rules are imperfect** (~88% field agreement on the audit
  set). Extraction errs toward omission (a missed criterion makes the engine
  say "eligible" too broadly, never "not eligible" wrongly on a stated
  criterion) — but users must still verify on the official page, which every
  answer links.
- Rules coverage is partial: Central schemes prioritized first; remaining
  State schemes pending the full extraction backfill.
- `category` mixes social category (SC/ST/OBC/EWS) with economic status
  (BPL) in one field; a user who is both SC and BPL should state the one the
  scheme asks about.
- Occupation matching is exact-token (a "mason" won't match "construction
  worker" yet).
- documents-required and FAQ endpoints not yet scraped; English only; small
  local models occasionally produce malformed actions (handled by JSON
  repair + code guards + non-LLM fallback).
