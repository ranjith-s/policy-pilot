"""
refresh.py — One-command data refresh: keep the knowledge base in sync with
the myScheme portal without redoing work that hasn't changed.

Steps (each skippable):
  1. rescrape        portal APIs -> raw jsons            (--skip-scrape)
  2. prepare         raw jsons -> data/rag_corpus.json
  3. diff            content_hash per scheme vs the extraction checkpoint
  4. extract         LLM re-extracts ONLY new/changed schemes
  5. merge           rewrite data/scheme_rules.csv (manual rows always win)
  6. embed           rebuild the semantic index if anything changed
  7. test            python tests/test_engine.py must stay green

Weekly cadence is plenty — schemes rarely change. Run before demos.

Usage:
    python pipeline/refresh.py --skip-scrape          # from existing raw jsons
    python pipeline/refresh.py --raw-dir data_raw     # after a fresh scrape
    python pipeline/refresh.py --model qwen3:4b-instruct --max-extract 50
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data"
sys.path.insert(0, str(SRC))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run(cmd, **kw):
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], **kw)
    if r.returncode != 0:
        sys.exit(f"refresh aborted: step failed with exit {r.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None,
                    help="dir with raw_schemes.json + raw_scheme_details.json "
                         "(default: scrape into data_raw/)")
    ap.add_argument("--skip-scrape", action="store_true",
                    help="reuse existing raw jsons in --raw-dir")
    ap.add_argument("--skip-embed", action="store_true")
    ap.add_argument("--model", default="qwen3:4b-instruct",
                    help="LLM for re-extracting changed schemes")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--max-extract", type=int, default=200,
                    help="safety cap: abort if more schemes changed than this "
                         "(big waves belong on the big GPU, run manually)")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else ROOT / "data_raw"

    # 1. scrape ------------------------------------------------------------
    if not args.skip_scrape:
        run([sys.executable, ROOT / "pipeline" / "rescrape.py",
             "--outdir", raw_dir])
    for f in ("raw_schemes.json", "raw_scheme_details.json"):
        if not (raw_dir / f).exists():
            sys.exit(f"missing {raw_dir / f} — run without --skip-scrape "
                     f"or point --raw-dir at the scrape output")

    # 2. prepare corpus ------------------------------------------------------
    run([sys.executable, ROOT / "pipeline" / "prepare_scheme_data.py",
         "--schemes", raw_dir / "raw_schemes.json",
         "--details", raw_dir / "raw_scheme_details.json",
         "--outdir", DATA])
    # prepare writes a blank scheme_rules_template.csv — informational only;
    # NEVER copy it over scheme_rules.csv (that wipes annotations)

    # 3. diff ----------------------------------------------------------------
    from extract_rules import _load_corpus, _done_hashes, content_hash, _manual_rows
    corpus = _load_corpus()
    done = _done_hashes()
    manual = set(_manual_rows())
    new = [sid for sid in corpus
           if sid not in done and sid not in manual
           and corpus[sid].get("eligibility_text")]
    changed = [sid for sid, h in done.items()
               if sid in corpus and h and h != content_hash(corpus[sid])]
    removed = [sid for sid in done if sid not in corpus]
    print(f"diff: {len(new)} new, {len(changed)} changed, {len(removed)} removed "
          f"(of {len(corpus)} schemes)")

    # 4. extract only what changed ------------------------------------------
    n_todo = len(new) + len(changed)
    if n_todo > args.max_extract:
        sys.exit(f"{n_todo} schemes need extraction (> --max-extract "
                 f"{args.max_extract}). Run src/extract_rules.py extract on "
                 f"the big GPU, then re-run refresh.")
    if n_todo:
        run([sys.executable, SRC / "extract_rules.py", "extract",
             "--model", args.model, "--host", args.host])

    # 5. merge ----------------------------------------------------------------
    run([sys.executable, SRC / "extract_rules.py", "merge"])

    # 6. embeddings ------------------------------------------------------------
    if not args.skip_embed and (new or changed or removed):
        run([sys.executable, SRC / "embeddings.py", "build"])
    elif not (new or changed or removed):
        print("no corpus changes — embeddings untouched")

    # 7. tests -------------------------------------------------------------------
    run([sys.executable, ROOT / "tests" / "test_engine.py"])

    report = {"new": len(new), "changed": len(changed), "removed": len(removed),
              "corpus_size": len(corpus)}
    (DATA / "refresh_report.json").write_text(json.dumps(report, indent=2))
    print(f"refresh complete: {report}")


if __name__ == "__main__":
    main()
