"""
overnight_gpu_extract.py — one-night big-model rule re-extraction on a
borrowed GPU box, then a safe merge back on the laptop.

ON THE GPU MACHINE (needs: python3, Ollama with the model pulled,
and this repo's  src/ + pipeline/ + data/rag_corpus.json + data/scheme_rules.csv):

    ollama pull qwen3.5:27b
    nohup python3 pipeline/overnight_gpu_extract.py run > overnight.log 2>&1 &
    tail -f overnight.log        # watch progress; safe to disconnect ssh

  Phases (each logged, each resumable — rerunning skips finished work):
    0  preflight   Ollama reachable, model present, data files found
    1  gold audit  re-extract the 5 hand-curated gold schemes, compare
                   field-by-field vs manual truth; ABORTS the night if
                   agreement is below --min-gold (default 80%)
    2  recover     schemes whose current extraction found NO structured
                   fields (engine can't check them today) — biggest win first
    3  upgrade     re-extract every other non-manual scheme, Central first
  Output: data/rules_llm_extracted.27b.jsonl  (+ overnight.log)

BACK ON THE LAPTOP next morning (after copying the .27b.jsonl into data/):

    python pipeline/overnight_gpu_extract.py finalize
    python tests/test_engine.py
    python src/extract_rules.py review --sample 10

  finalize backs up scheme_rules.csv and the old checkpoint, concatenates
  old + new checkpoints (new rows win — schemes the night didn't reach
  keep their old extraction), and rebuilds scheme_rules.csv via merge.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from extract_rules import (   # noqa: E402  (stdlib-only module)
    CHECKPOINT, CORPUS_PATH, RULES_CSV, OllamaExtract, _load_corpus,
    _manual_rows, _validate, content_hash, cmd_merge,
)

NEW_CHECKPOINT = ROOT / "data" / "rules_llm_extracted.27b.jsonl"
GOLD_IDS = ["sui", "pmsby", "sfava", "famdpwog", "rgisfm"]
STRUCT_COLS = ["age_min", "age_max", "income_max_annual", "gender", "category",
               "occupation", "marital_status", "requires_bank_account", "land_owner"]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- preflight --
def preflight(args):
    log("phase 0: preflight")
    for path in (CORPUS_PATH, RULES_CSV):
        if not path.exists():
            sys.exit(f"FATAL: {path} not found — copy data/ from the laptop first")
    try:
        with urllib.request.urlopen(f"{args.host}/api/tags", timeout=10) as r:
            models = [m["name"] for m in json.loads(r.read())["models"]]
    except Exception as e:
        sys.exit(f"FATAL: Ollama not reachable at {args.host} ({e}). Start it first.")
    if not any(m.split(":latest")[0] == args.model or m == args.model for m in models):
        sys.exit(f"FATAL: model '{args.model}' not pulled. Available: {models}\n"
                 f"Run: ollama pull {args.model}")
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
        log(f"  GPU: {gpu or 'nvidia-smi gave no output'}")
    except Exception:
        log("  GPU: nvidia-smi not available (continuing anyway)")
    log(f"  Ollama OK, model '{args.model}' present, data files found")


# --------------------------------------------------------------- gold audit --
def _rules_rows():
    import csv
    with open(RULES_CSV, newline="", encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def _norm(field, v):
    v = str(v or "").strip().lower()
    if field in ("age_min", "age_max", "income_max_annual") and v:
        try:
            return str(int(float(v.replace(",", ""))))
        except ValueError:
            return v
    if field == "category":
        return ",".join(sorted(p.strip() for p in v.split(",") if p.strip()))
    return v


def gold_audit(args, llm, corpus):
    log("phase 1: gold audit — re-extracting the 5 hand-curated schemes")
    truth = {sid: r for sid, r in _rules_rows().items() if sid in GOLD_IDS}
    if len(truth) < len(GOLD_IDS):
        sys.exit(f"FATAL: gold rows missing from {RULES_CSV}: "
                 f"{sorted(set(GOLD_IDS) - set(truth))}")
    # land_owner is never extracted by the LLM (too error-prone) — don't
    # count it against the model
    fields = [c for c in STRUCT_COLS if c != "land_owner"]
    match = total = 0
    for sid in GOLD_IDS:
        d = corpus[sid]
        row = _validate(llm.extract(d["scheme_name"], d["eligibility_text"],
                                    d.get("exclusions", "")))
        diffs = []
        for f in fields:
            total += 1
            if _norm(f, row.get(f)) == _norm(f, truth[sid].get(f)):
                match += 1
            else:
                diffs.append(f"{f}: got {row.get(f)!r} want {truth[sid].get(f)!r}")
        log(f"  {sid}: {len(fields) - len(diffs)}/{len(fields)}"
            + (f"  [{'; '.join(diffs)}]" if diffs else "  perfect"))
    pct = 100.0 * match / total
    log(f"  gold agreement: {pct:.0f}% (baseline with the 4B model was 88%)")
    if pct < args.min_gold:
        sys.exit(f"ABORT: {pct:.0f}% is below --min-gold {args.min_gold}%. "
                 "The big model is not beating the baseline — investigate "
                 "before spending the night.")
    return pct


# --------------------------------------------------------------- extraction --
def _new_done():
    if not NEW_CHECKPOINT.exists():
        return set()
    with open(NEW_CHECKPOINT, encoding="utf-8") as f:
        return {json.loads(l)["id"] for l in f if l.strip()}


def run_batch(label, docs, llm):
    """Extract docs sequentially into NEW_CHECKPOINT with rate/ETA logs.
    Resume-safe (skips ids already in the new checkpoint)."""
    done = _new_done()
    docs = [d for d in docs if d["id"] not in done]
    log(f"{label}: {len(docs)} schemes to extract")
    if not docs:
        return
    t0, ok, fail, consec_fail = time.time(), 0, 0, 0
    with open(NEW_CHECKPOINT, "a", encoding="utf-8") as ckpt:
        for i, d in enumerate(docs, 1):
            try:
                row = _validate(llm.extract(d["scheme_name"], d["eligibility_text"],
                                            d.get("exclusions", "")))
                row["id"] = d["id"]
                row["content_hash"] = content_hash(d)
                ckpt.write(json.dumps(row, ensure_ascii=False) + "\n")
                ckpt.flush()
                ok += 1
                consec_fail = 0
            except Exception as e:
                fail += 1
                consec_fail += 1
                log(f"  !! {d['id']}: {e!r}")
                if consec_fail >= 20:
                    sys.exit("ABORT: 20 consecutive failures — Ollama has "
                             "probably died. Fix it and rerun (progress is saved).")
            if i % 25 == 0 or i == len(docs):
                rate = (time.time() - t0) / i
                eta_h = rate * (len(docs) - i) / 3600
                log(f"  {label}: {i}/{len(docs)}  ok={ok} fail={fail}  "
                    f"{rate:.1f}s/scheme  ~{eta_h:.1f}h left")
    log(f"{label} finished: {ok} extracted, {fail} failed, "
        f"{(time.time() - t0) / 3600:.1f}h elapsed")


def cmd_run(args):
    log(f"=== overnight extraction run (model={args.model}) ===")
    preflight(args)
    corpus = _load_corpus()
    llm = OllamaExtract(model=args.model, host=args.host)
    gold_audit(args, llm, corpus)
    if args.gold_only:
        log("--gold-only set: stopping after the audit.")
        return

    manual = set(_manual_rows())
    rules = _rules_rows()
    eligible_docs = {sid: d for sid, d in corpus.items()
                     if sid not in manual and d.get("eligibility_text")}

    # phase 2: schemes the current CSV can't check at all (no structured cols)
    no_struct = [d for sid, d in sorted(eligible_docs.items())
                 if not any((rules.get(sid, {}).get(c) or "").strip()
                            for c in STRUCT_COLS)]
    log(f"phase 2: recover engine-uncheckable schemes")
    run_batch("recover", no_struct, llm)

    # phase 3: quality upgrade for everything else, Central-level first
    rest = [d for sid, d in sorted(eligible_docs.items())
            if any((rules.get(sid, {}).get(c) or "").strip() for c in STRUCT_COLS)]
    rest.sort(key=lambda d: (0 if "central" in (d.get("level") or "").lower() else 1,
                             d["id"]))
    log(f"phase 3: re-extract remaining schemes with the big model")
    run_batch("upgrade", rest, llm)

    log("=== ALL DONE ===")
    log(f"copy back to the laptop:  data/{NEW_CHECKPOINT.name}")
    log("then on the laptop:  python pipeline/overnight_gpu_extract.py finalize")


# ----------------------------------------------------------------- finalize --
def cmd_finalize(args):
    new = Path(args.new) if args.new else NEW_CHECKPOINT
    if not new.exists():
        sys.exit(f"FATAL: {new} not found — copy it back from the GPU machine "
                 f"(or pass --new <path>)")
    n_new = sum(1 for l in open(new, encoding="utf-8") if l.strip())
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for f in (RULES_CSV, CHECKPOINT):
        if f.exists():
            backup = f.with_suffix(f.suffix + f".{stamp}.bak")
            shutil.copy2(f, backup)
            log(f"backup: {backup.name}")

    # old checkpoint first, new lines appended after — merge keeps the LAST
    # line per id, so big-model rows win and unreached schemes keep the old
    # extraction
    with open(CHECKPOINT, "a", encoding="utf-8") as out, \
         open(new, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.write(line.rstrip("\n") + "\n")
    log(f"appended {n_new} big-model rows to {CHECKPOINT.name} (they override)")
    cmd_merge(argparse.Namespace())
    log("done. Now run:  python tests/test_engine.py")
    log("      and:      python src/extract_rules.py review --sample 10")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="overnight extraction (on the GPU machine)")
    r.add_argument("--model", default="qwen3.5:27b")
    r.add_argument("--host", default="http://localhost:11434")
    r.add_argument("--min-gold", type=float, default=80.0,
                   help="abort if gold agreement %% is below this")
    r.add_argument("--gold-only", action="store_true",
                   help="run the audit and stop (quick model sanity check)")
    f = sub.add_parser("finalize", help="merge results back (on the laptop)")
    f.add_argument("--new", default=None,
                   help=f"path to the copied-back jsonl (default data/{NEW_CHECKPOINT.name})")
    args = p.parse_args()
    {"run": cmd_run, "finalize": cmd_finalize}[args.cmd](args)


if __name__ == "__main__":
    main()
