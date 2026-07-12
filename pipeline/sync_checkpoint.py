"""
sync_checkpoint.py — Post-annotation helper.

Sanitizes data/rules_llm_extracted.jsonl (drops any truncated trailing line
from a mid-write scp), reports row counts, then reminds you of the next steps.

Usage:
    python pipeline/sync_checkpoint.py
"""

import json
import sys
from pathlib import Path

CKPT = Path(__file__).resolve().parent.parent / "data" / "rules_llm_extracted.jsonl"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

lines = [l for l in CKPT.read_text(encoding="utf-8").splitlines() if l.strip()]
good, ids, hashless = [], set(), 0
for l in lines:
    try:
        r = json.loads(l)
    except Exception:
        continue                      # drop truncated/corrupt line
    good.append(l)
    ids.add(r["id"])
    if not r.get("content_hash"):
        hashless += 1

CKPT.write_text("\n".join(good) + "\n", encoding="utf-8")
print(f"{len(good)} valid rows ({len(lines) - len(good)} dropped), "
      f"{len(ids)} unique schemes, {hashless} still hashless")
print("next:  python src/extract_rules.py merge  &&  python tests/test_engine.py")
