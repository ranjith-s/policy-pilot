# Public Scheme Eligibility Assistant

An agentic AI assistant that helps citizens discover Indian government welfare
schemes, check their eligibility, and know exactly what documents and steps
come next — built on the [myScheme portal](https://www.myscheme.gov.in) corpus.

**Core design principle:** the LLM never decides eligibility. Every verdict
comes from a **deterministic rules engine**, so results are reproducible,
auditable, and hallucination-free. The LLM's two jobs are (a) driving the
conversation as a ReAct agent and (b) building the knowledge base offline by
extracting structured rules from official eligibility text.

## Key numbers

| What | Count |
|---|---|
| Schemes in knowledge base (full myScheme scrape) | 4,682 |
| Schemes with engine-checkable eligibility rules | 3,810 (extraction complete; the other 867 have free-text-only criteria the engine can't check) |
| Schemes with official document checklists | 4,662 |
| Schemes with official FAQs | 4,664 |
| Rule-extraction accuracy vs hand-curated gold rows | 88% field agreement |
| LLM-guessed eligibility verdicts | 0 (engine-only, enforced in code) |

## Architecture

Two conversation modes share the same engine, retrieval, and data — a
deliberate design comparison for the Agentic AI course:

### Guided funnel (default) — `src/funnel.py`

The conversation **policy is code**; the LLM's only job is extracting facts
from free text. Per user turn: exactly **one LLM call** (fact + topic-word
extraction) → deterministic engine over all rules (~40 ms) →
**relevance-blended ranking** (semantic similarity of the user's own words to
each scheme, blended 65/35 with rule specificity; keyword fallback if the
embedding service is down) → templated answer showing *confirmed eligible* +
*likely candidates with their missing facts* + the single
highest-information follow-up question (chosen from what blocks the user's
TOP candidates). `more` pages deeper with **zero** LLM calls. Result: ~8 s
per turn on a 4 GB GPU, wide-net-then-narrow funnel guaranteed by
construction.

### Free ReAct agent (`--react`) — `src/agent.py`

The LLM picks a tool each step (the classic agent loop below). More flexible,
but a small local model may skip follow-ups or write slow, long answers —
measured failure modes that motivated the funnel (see the trace logs).

```
User (CLI or Streamlit UI)
   │
   ▼
ReAct Agent Loop (src/agent.py)        ← LLM decides ONE action per step:
   │   {"thought", "action", "action_input"}
   ▼
Toolbox (src/tools.py)
   ├── search_schemes          hybrid retrieval: metadata filter → semantic
   │                           (cosine over precomputed embeddings) blended
   │                           with keyword score; keyword-only fallback
   ├── run_eligibility_check   → engine.py (deterministic, source of truth)
   ├── get_scheme_details      benefits, application steps, OFFICIAL document
   │                           checklist + top FAQs (quoted, not paraphrased)
   ├── update_profile          session memory — batched: saves ALL facts from
   │                           a message in ONE LLM step
   ├── ask_user                human-in-the-loop question (ends the turn)
   └── final_answer            ends the turn (must pass the guards below)
   │
   ▼
Deterministic Rules Engine (src/engine.py)
   reads data/scheme_rules.csv → eligible / partial / not_eligible per scheme,
   with reasons, missing fields, and a suggested next question
```

### Two-tier eligibility data

- **Tier 1 — metadata** (state, level, category): parsed directly from the
  scrape; used as a hard filter in retrieval.
- **Tier 2 — structured rules** (`data/scheme_rules.csv`): age_min/max,
  income_max_annual, gender, category (SC/ST/OBC/EWS/minority/BPL +
  `female_any_category` token), occupation, marital_status,
  requires_bank_account, land_owner, plus free-text `other_conditions`.
  Each row carries `rules_source`: `manual` (5 hand-curated gold rows) or
  `llm` (extracted by `src/extract_rules.py`). Rows with **no structured
  criteria are ignored by the engine** — a scheme the engine can't actually
  check can never be marked "eligible" (this guard exists because a blank
  template once overwrote the CSV and made everyone eligible for everything).

### Guardrails (enforced in code, not just prompt)

1. **Verdict guard** — a `final_answer` claiming eligibility is rejected
   unless the engine's latest run returned that scheme as eligible; any
   non-eligible scheme it names must carry a qualifier ("possibly eligible",
   "more info needed") on the same line.
2. **Engine-first guard** — with profile facts on file, a `final_answer` is
   rejected unless `run_eligibility_check` was called **in the current turn**
   (stale verdicts from earlier turns don't count).
3. **Loop cap** — max 8 tool calls per user turn.
4. **Question cap** — max 2 `ask_user` rounds per conversation, then
   best-effort answer.
5. **JSON repair** — malformed LLM output gets one repair attempt; bare-string
   `action_input` is coerced into the expected dict shape; after that a
   non-LLM fallback answer is built from the engine's last results (a demo
   can never dead-end).
6. **Trace log** — every thought/action/observation appended to
   `logs/agent_trace.jsonl` with per-step timings.

### Where the LLM is used (and why)

| Role | When | Why an LLM |
|---|---|---|
| Conversation agent | query time | understands free-text ("I'm a widow in Delhi"), picks tools, asks the right follow-up |
| Rule extraction | offline, one-time + refresh | converts 4,682 schemes' official eligibility text into structured CSV rows — impossible to hand-annotate |
| **Not** eligibility verdicts | never | verdicts must be deterministic, auditable, and instant |

## Data pipeline

```
myScheme portal APIs (list + detail + documents + faqs)
   │  pipeline/rescrape.py, pipeline/scrape_extras.py   (rate-limited 1 req/s,
   │                                                     resumable, key via env)
   ▼
raw JSONs ──pipeline/prepare_scheme_data.py──► data/rag_corpus.json (4,682 docs)
   │                                              │
   │  src/extract_rules.py extract                │  pipeline/scrape_extras.py enrich
   │  (LLM reads eligibility_text → structured    │  (official documents_required +
   │   fields; conservative: null unless          │   faqs merged into corpus)
   │   explicit; validated against controlled     │
   │   vocabularies; checkpointed/resumable)      │
   ▼                                              ▼
data/rules_llm_extracted.jsonl ──merge──► data/scheme_rules.csv
                                          (manual rows always win)
   │
   ▼
src/embeddings.py build ──► data/embeddings.npy (4,682 × 768, nomic-embed-text,
                            cosine search via numpy — no vector DB needed)
```

**Weekly refresh** (`python pipeline/refresh.py`): rescrape → rebuild corpus →
**content-hash diff** → LLM re-extracts *only new/changed schemes* → merge →
re-embed → engine tests must pass. A typical week changes a handful of
schemes, so refresh costs minutes. `data/refresh_report.json` records each run.

**Rule extraction quality** is measured, not assumed: extracting the 5
hand-curated gold schemes and comparing field-by-field gives **88% agreement**;
disagreements were used to improve the prompt (e.g. the applicant-vs-dependent
distinction, the `female_any_category` token). Audit any sample yourself:
`python src/extract_rules.py review --sample 10`.

**Document-portfolio insight** (`python pipeline/scrape_extras.py insights`):
aggregating official checklists shows a small set of documents (bank
passbook 56%, Aadhaar 50%, ration card 48%, photo 46%) unlocks roughly half
of all schemes — the agent can tell users which documents to obtain first.

## LLM backends

All backends implement one interface — `.chat(messages) -> str` (~40 lines
each, stdlib only, no framework needed):

| Backend | Flag / UI option | Default model | Key |
|---|---|---|---|
| Ollama (local) | default | `qwen3:4b-instruct` | — |
| Google Gemini | `--gemini` | `gemini-2.5-flash` | `GEMINI_API_KEY` env |
| OpenAI / ChatGPT | `--chatgpt` | `gpt-4o-mini` | `OPENAI_API_KEY` env |
| MockLLM | `--mock` | scripted | — |

Notes baked into the clients: JSON-constrained output on all three real
backends; `think: false` for thinking-mode models (with 400-fallback for
models that reject it); `num_ctx` pinned to 8192 (hosts with a 32k default
allocate a KV cache that can overflow VRAM and silently spill to CPU);
retries on transient errors; reasoning models (`o*`/`gpt-5*`) get no custom
temperature. **Never hardcode API keys — env vars only.**

## Latency design

One user turn = several *sequential* LLM calls (that's the ReAct trade-off),
so the app optimizes call count and visibility:

- **Batched profile updates**: all facts from a message saved in ONE
  `update_profile` call (was 3+ calls).
- **Compact observations**: the engine result sent to the LLM is capped
  (top 10 eligible + top 5 partial + counts); the agent keeps the full
  result internally for its guards.
- **Fail-fast retrieval**: if the embedding service is down, search falls
  back to keyword instantly instead of retry backoff.
- **Warm caches**: corpus (37 MB) and semantic index preloaded at app start.
- **Per-step timing**: every step shows its seconds in the CLI trace and the
  UI status panel, so slowness is diagnosable, not mysterious.

## Run

```bash
pip install -r requirements.txt        # numpy + streamlit; Ollama separate

# Web UI (recommended) — http://localhost:8501
streamlit run app/app.py

# CLI (guided funnel by default)
python src/main.py --show-trace        # local Ollama (pull qwen3:4b-instruct)
python src/main.py --react             # free ReAct agent mode (comparison)
python src/main.py --gemini            # cloud (set GEMINI_API_KEY)
python src/main.py --mock --show-trace # no LLM needed (scripted demo)
```

Try: *"I'm a 65 year old artist with annual income of 40000 rupees"* — watch
the agent save facts, search, run the engine, and ask the engine-suggested
follow-up. CLI commands: `profile`, `reset`, `quit`.

### Rebuilding data from scratch

```bash
# 1. scrape the portal (set MYSCHEME_API_KEY — see pipeline/rescrape.py header)
python pipeline/rescrape.py --outdir data_raw
python pipeline/prepare_scheme_data.py --schemes data_raw/raw_schemes.json \
       --details data_raw/raw_scheme_details.json --outdir data

# 2. extract eligibility rules (resumable; --limit N to run in sessions;
#    --priority-central does nationally-applicable schemes first)
python src/extract_rules.py extract --model qwen3.5:27b --priority-central
python src/extract_rules.py merge
python src/extract_rules.py review --sample 10     # manual QA

# 3. documents + FAQs (parallel streams, resumable)
python pipeline/scrape_extras.py scrape
python pipeline/scrape_extras.py enrich
python pipeline/scrape_extras.py insights

# 4. semantic index
python src/embeddings.py build          # needs `ollama pull nomic-embed-text`

# 5. verify
python tests/test_engine.py
```

Use the largest instruction-tuned model your GPU fits for step 2 — quality of
extraction directly bounds verdict quality. Everything is checkpointed:
interrupt and re-run freely; already-done schemes are skipped via content hash.

## Tests / validation

`python tests/test_funnel.py` — guided-funnel policy validation (no LLM
service needed): wide-net candidate surfacing, farmer-relevance ranking,
exactly-one-LLM-call-per-turn, zero-call `more` pagination, narrowing on
follow-up answers, graceful extraction failure.

`python tests/test_engine.py` — persona-based validation (no pytest needed):
7 hand-built personas against the 5 gold schemes (19 assertions) + personas
against LLM-extracted schemes (widow pension eligible/not-eligible by gender,
income-capped social security) + next-question logic. LLM-scheme tests
auto-skip if those schemes aren't merged yet.

## Repository structure

```
├── app/app.py                  Streamlit chat UI (backends, live steps, stats)
├── src/
│   ├── main.py                 CLI entry point
│   ├── agent.py                ReAct loop + guardrails + trace log
│   ├── tools.py                toolbox + retrieval + observation compaction
│   ├── engine.py               deterministic eligibility engine
│   ├── llm.py                  Ollama / Gemini / OpenAI / Mock backends
│   ├── embeddings.py           build + query semantic index
│   └── extract_rules.py        LLM rule extraction (extract/review/merge)
├── pipeline/
│   ├── rescrape.py             portal list+detail scraper (rate-limited)
│   ├── scrape_extras.py        documents+FAQs scraper / enrich / insights
│   ├── prepare_scheme_data.py  raw scrape → rag_corpus.json
│   └── refresh.py              one-command weekly refresh orchestrator
├── data/
│   ├── rag_corpus.json         4,682 enriched scheme documents
│   ├── scheme_rules.csv        structured rules (manual + llm, provenance col)
│   ├── rules_llm_extracted.jsonl  extraction checkpoint (resumable)
│   ├── embeddings.npy + embedding_ids.json   semantic index
│   ├── scheme_object_ids.json  slug → portal _id map (for v6 APIs)
│   └── raw_documents_required.json / raw_faqs.json   scrape caches
├── tests/test_engine.py        persona-based engine validation
└── logs/agent_trace.jsonl      full agent step trace (gitignored)
```

## Limitations / responsible use

Results are **indicative only** — every answer links the official myScheme
portal and says so. Known limitations:

- LLM-extracted rules are ~88% accurate on the audit set; extraction errs
  toward omission (a missed criterion over-includes, never wrongly excludes
  on a stated criterion). Users must verify on the official page.
- 867 schemes have free-text-only eligibility criteria (institutional /
  procedural conditions) that don't map to structured fields — the engine
  can't verdict these; the agent may only describe them (`rules_available:
  false` in search results).
- `category` mixes social category (SC/ST/OBC/EWS) with economic status
  (BPL) in one field.
- Occupation matching is exact-token ("mason" won't match "construction
  worker" yet).
- English only; small local models occasionally emit malformed actions
  (handled by repair + guards + non-LLM fallback).
- The scrapers are polite (1 req/s, resumable) — never parallel-hammer the
  government API.

## Future improvements

- Periodic `refresh.py` runs to keep the corpus and rules current.
- Fuzzy occupation/category matching; separate BPL from social category.
- Query-time LLM "soft check" (clearly labeled non-verdict) for schemes
  without structured rules.
- Multilingual answers; document-gap checklist interaction ("I have Aadhaar
  and a passbook — what am I missing for scheme X?").
- Port the hand-rolled agent loop to LangGraph as a learning exercise — the
  tools, engine, and guards map 1:1 onto nodes, conditional edges, and
  interrupts.
