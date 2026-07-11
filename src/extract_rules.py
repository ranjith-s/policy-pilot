"""
extract_rules.py — LLM-assisted extraction of structured eligibility rules.

Reads each scheme's official eligibility_text from data/rag_corpus.json and
asks a local LLM to fill the SAME structured columns used by the hand-annotated
rows in scheme_rules.csv. The engine stays deterministic at query time; the
LLM only helps BUILD the knowledge base, offline, once.

Design:
  * Conservative by construction: the model must leave a field null unless the
    text states it explicitly, outputs are validated/coerced in Python, and
    anything that doesn't fit the schema is demoted to other_conditions text.
  * Checkpointed: every result is appended to data/rules_llm_extracted.jsonl;
    re-running skips schemes already done (safe to interrupt).
  * Manual annotations always win: merge never overwrites a manual row.

Usage:
    python src/extract_rules.py extract --sample 40   # seeded random pilot
    python src/extract_rules.py extract               # all remaining schemes
    python src/extract_rules.py review  --sample 10   # eyeball extractions vs source
    python src/extract_rules.py merge                 # rewrite scheme_rules.csv
"""

import argparse
import csv
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Windows consoles default to cp1252, which can't print ₹ etc.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_PATH = DATA_DIR / "rag_corpus.json"
RULES_CSV = DATA_DIR / "scheme_rules.csv"
CHECKPOINT = DATA_DIR / "rules_llm_extracted.jsonl"

OCCUPATIONS = [
    "farmer", "fisherman", "student", "teacher", "faculty", "artist",
    "artisan", "weaver", "entrepreneur", "construction worker",
    "journalist", "advocate", "sportsperson", "ex-serviceman",
]
CATEGORIES = ["sc", "st", "obc", "ews", "minority", "bpl", "female_any_category"]
MARITAL = ["widow", "married", "unmarried", "divorced"]

SYSTEM = ("You extract structured eligibility rules for Indian government "
          "welfare schemes. Reply with ONLY one JSON object, no other text.")

PROMPT = """Scheme: {name}

Eligibility text:
{elig}

Exclusions:
{excl}

Extract into exactly this JSON shape (use null when not stated):
{{"age_min": null, "age_max": null, "income_max_annual": null, "gender": null, "category": null, "occupation": null, "marital_status": null, "requires_bank_account": null, "other_conditions": ""}}

Rules:
- Fill a field ONLY when the text states it explicitly. Never guess.
- All fields describe the person APPLYING. If a scheme helps someone's dependents (e.g. assistance for a widow's daughter's marriage), the applicant is the widow — extract HER gender/marital status.
- age_min / age_max: integer years for the primary applicant.
- income_max_annual: maximum annual income in plain rupees (1 lakh = 100000, 1 crore = 10000000).
- gender: "male" or "female" ONLY if EVERY applicant must be that gender. If a scheme is open to both genders (even at different subsidy rates), leave null.
- category: comma-separated subset of {cats} — ONLY if the scheme is RESTRICTED to those groups (different benefit rates for different groups is NOT a restriction). Special token: if SC/ST (or similar) applicants qualify AND women of ANY category also qualify, include female_any_category.
- occupation: one of {occs} — ONLY if the scheme is restricted to that occupation.
- marital_status: one of {mars} — ONLY if restricted.
- requires_bank_account: true if the applicant must hold a bank account (including auto-debit consent or DBT payout to own account).
- other_conditions: remaining important criteria in under 200 characters (residency, registration, disability %, etc). "" if none.
"""


class OllamaExtract:
    def __init__(self, model="qwen3:4b-instruct", host="http://localhost:11434"):
        self.model, self.host = model, host

    def extract(self, name, elig, excl, retries=2):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT.format(
                    name=name, elig=elig[:1500], excl=excl[:400] or "(none)",
                    cats=", ".join(CATEGORIES), occs=", ".join(OCCUPATIONS),
                    mars=", ".join(MARITAL))},
            ],
            "stream": False,
            "format": "json",
            "keep_alive": "30m",
            # thinking models (qwen3.5 etc.) can reason for minutes per
            # request before emitting JSON — disable it, we want extraction
            "think": False,
            # small num_ctx: extraction prompts are ~700 tokens; the host's
            # default (32k) allocates a KV cache that can overflow VRAM
            "options": {"temperature": 0.0, "num_ctx": 4096},
        }
        for attempt in range(retries + 1):
            req = urllib.request.Request(
                f"{self.host}/api/chat",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    return json.loads(json.loads(resp.read())["message"]["content"])
            except urllib.error.HTTPError as e:
                # older servers / non-thinking models reject "think"
                if e.code == 400 and "think" in payload:
                    payload.pop("think")
                    continue
                if attempt == retries:
                    raise
                time.sleep(2 ** attempt)
            except Exception:
                if attempt == retries:
                    raise
                time.sleep(2 ** attempt)


def _validate(raw):
    """Coerce/validate LLM output; anything off-schema is dropped or demoted
    to other_conditions. Returns a dict of CSV-ready string values."""
    out = {}
    extras = []

    def num(key, as_int=False):
        v = raw.get(key)
        try:
            v = float(str(v).replace(",", ""))
            out[key] = str(int(v)) if as_int or v == int(v) else str(v)
        except (TypeError, ValueError):
            out[key] = ""

    num("age_min", as_int=True)
    num("age_max", as_int=True)
    num("income_max_annual")

    g = str(raw.get("gender") or "").strip().lower()
    out["gender"] = g if g in ("male", "female") else ""

    cats = [c.strip().lower() for c in str(raw.get("category") or "").split(",")]
    cats = [c for c in cats if c in CATEGORIES]
    out["category"] = ",".join(cats)

    occ = str(raw.get("occupation") or "").strip().lower()
    if occ in OCCUPATIONS:
        out["occupation"] = occ
    else:
        out["occupation"] = ""
        if occ and occ not in ("null", "none", "any"):
            extras.append(f"for: {occ}")

    ms = str(raw.get("marital_status") or "").strip().lower()
    out["marital_status"] = ms if ms in MARITAL else ""

    out["requires_bank_account"] = "yes" if raw.get("requires_bank_account") is True else ""
    out["land_owner"] = ""   # not extracted (too error-prone from text)

    oc = str(raw.get("other_conditions") or "").strip()
    if oc.lower() in ("null", "none"):
        oc = ""
    out["other_conditions"] = "; ".join(filter(None, [oc[:250]] + extras))
    return out


def _load_corpus():
    with open(CORPUS_PATH, encoding="utf-8") as f:
        return {d["id"]: d for d in json.load(f)}


def content_hash(doc):
    """Fingerprint of the text the rules were extracted from. A changed hash
    on refresh means the portal updated the scheme -> re-extract it."""
    basis = (doc.get("scheme_name", "") + "\x1f"
             + doc.get("eligibility_text", "") + "\x1f"
             + doc.get("exclusions", ""))
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def _done_hashes():
    """id -> content_hash of already-extracted schemes (checkpoint)."""
    if not CHECKPOINT.exists():
        return {}
    out = {}
    with open(CHECKPOINT, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r.get("content_hash", "")
    return out


def _manual_rows():
    """Existing manual annotations (rules_source blank or 'manual')."""
    if not RULES_CSV.exists():
        return {}
    with open(RULES_CSV, newline="", encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)
                if (r.get("rules_source") or "manual") == "manual"
                and any((r.get(c) or "").strip() for c in (
                    "age_min", "age_max", "income_max_annual", "gender",
                    "category", "occupation", "marital_status",
                    "requires_bank_account", "land_owner"))}


def cmd_extract(args):
    corpus = _load_corpus()
    done = _done_hashes()
    manual = set(_manual_rows())
    todo = []
    for sid, d in sorted(corpus.items()):
        if sid in manual or not d.get("eligibility_text"):
            continue
        # skip only if already extracted AND source text unchanged
        if sid in done and done[sid] == content_hash(d):
            continue
        todo.append(d)
    if args.priority_central:
        # Central schemes apply to users in any state — extract them first
        todo.sort(key=lambda d: (0 if "central" in (d.get("level") or "").lower() else 1,
                                 d["id"]))
    if args.sample:
        rng = random.Random(42)   # seeded: pilot sample is reproducible
        todo = rng.sample(todo, min(args.sample, len(todo)))
    if args.limit:
        todo = todo[:args.limit]

    print(f"{len(todo)} schemes to extract "
          f"({len(done)} already done/manual, model={args.model})", flush=True)
    llm = OllamaExtract(model=args.model, host=args.host)
    t0 = time.time()
    with open(CHECKPOINT, "a", encoding="utf-8") as ckpt:
        for i, d in enumerate(todo, 1):
            try:
                raw = llm.extract(d["scheme_name"], d["eligibility_text"],
                                  d.get("exclusions", ""))
                row = _validate(raw)
            except Exception as e:
                print(f"  !! {d['id']}: {e!r}", flush=True)
                continue
            row["id"] = d["id"]
            row["content_hash"] = content_hash(d)
            ckpt.write(json.dumps(row, ensure_ascii=False) + "\n")
            ckpt.flush()
            if i % 10 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                eta_min = rate * (len(todo) - i) / 60
                print(f"  {i}/{len(todo)}  ({rate:.1f}s/scheme, ~{eta_min:.0f} min left)",
                      flush=True)
    print("done.")


def cmd_review(args):
    """Print extracted fields next to the source text for manual QA."""
    corpus = _load_corpus()
    rows = []
    with open(CHECKPOINT, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rng = random.Random(args.seed)
    for row in rng.sample(rows, min(args.sample, len(rows))):
        d = corpus.get(row["id"], {})
        print("=" * 78)
        print(d.get("scheme_name", row["id"]), f"[{row['id']}]")
        print("- source:", (d.get("eligibility_text") or "")[:400].replace("\n", " "))
        print("- extracted:", {k: v for k, v in row.items() if v and k != "id"})


def cmd_merge(args):
    """Rewrite scheme_rules.csv: manual rows first (never overwritten),
    then LLM-extracted rows, then blank rows for uncovered schemes."""
    corpus = _load_corpus()
    manual = _manual_rows()
    llm_rows = {}
    if CHECKPOINT.exists():
        with open(CHECKPOINT, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    llm_rows[r["id"]] = r   # later lines win (re-runs)

    cols = ["id", "scheme_name", "level", "states", "age_min", "age_max",
            "income_max_annual", "gender", "category", "occupation",
            "marital_status", "requires_bank_account", "land_owner",
            "other_conditions", "documents_required", "rules_source"]
    n_manual = n_llm = n_blank = 0
    with open(RULES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for sid, d in sorted(corpus.items()):
            base = {"id": sid, "scheme_name": d["scheme_name"],
                    "level": d["level"], "states": "|".join(d["states"])}
            if sid in manual:
                row = {**base, **{k: manual[sid].get(k, "") for k in cols[4:]},
                       "rules_source": "manual"}
                n_manual += 1
            elif sid in llm_rows:
                row = {**base, **llm_rows[sid], "documents_required": "",
                       "rules_source": "llm"}
                n_llm += 1
            else:
                row = base
                n_blank += 1
            w.writerow(row)
    print(f"wrote {RULES_CSV}: {n_manual} manual, {n_llm} llm, {n_blank} unannotated")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--sample", type=int, default=0, help="pilot: N random schemes")
    e.add_argument("--limit", type=int, default=0, help="stop after N schemes")
    e.add_argument("--priority-central", action="store_true",
                   help="extract Central-level schemes first")
    e.add_argument("--model", default="qwen3:4b-instruct")
    e.add_argument("--host", default="http://localhost:11434")
    r = sub.add_parser("review")
    r.add_argument("--sample", type=int, default=10)
    r.add_argument("--seed", type=int, default=7)
    sub.add_parser("merge")
    args = p.parse_args()
    {"extract": cmd_extract, "review": cmd_review, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
