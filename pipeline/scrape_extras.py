"""
scrape_extras.py — One-time scrape of the per-scheme documents-required and
FAQ APIs, then enrich data/rag_corpus.json in place.

Same etiquette as rescrape.py: single-threaded, rate-limited, resumable
(slugs already fetched are skipped), MYSCHEME_API_KEY from env.

Usage:
    # scrape only rule-covered schemes (fast, demo-critical):
    python pipeline/scrape_extras.py scrape --rules-only

    # scrape everything (one-time, ~2.5 h at 1 req/s):
    python pipeline/scrape_extras.py scrape

    # merge scraped docs/faqs into rag_corpus.json:
    python pipeline/scrape_extras.py enrich

    # print the document-portfolio insight:
    python pipeline/scrape_extras.py insights
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOCS_PATH = DATA / "raw_documents_required.json"
FAQS_PATH = DATA / "raw_faqs.json"

API_BASE = os.environ.get("MYSCHEME_API_BASE", "https://api.myscheme.gov.in")
API_KEY = os.environ.get("MYSCHEME_API_KEY", "")
# endpoint templates ({slug} substituted); override via env if the portal moves
DOCS_URL = os.environ.get(
    "MYSCHEME_DOCS_URL",
    API_BASE + "/schemes/v5/public/schemes/{slug}/documents?lang=en")
FAQS_URL = os.environ.get(
    "MYSCHEME_FAQS_URL",
    API_BASE + "/schemes/v5/public/schemes/{slug}/faqs?lang=en")
RATE_DELAY = 1.0

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _get(url):
    req = urllib.request.Request(url, headers={
        "x-api-key": API_KEY,
        "Accept": "application/json",
        "User-Agent": "scheme-agent-refresh/1.0 (student capstone)",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _rich_text(nodes):
    """Flatten myScheme rich-text JSON (nested type/children blocks) to lines."""
    out = []

    def walk(node):
        if isinstance(node, list):
            for n in node:
                walk(n)
        elif isinstance(node, dict):
            if "text" in node and node["text"].strip():
                out.append(node["text"].strip())
            walk(node.get("children", []))

    walk(nodes)
    return out


def _corpus():
    with open(DATA / "rag_corpus.json", encoding="utf-8") as f:
        return json.load(f)


def _rules_slugs():
    with open(DATA / "scheme_rules.csv", newline="", encoding="utf-8") as f:
        return {r["id"] for r in csv.DictReader(f)
                if (r.get("rules_source") or "").strip()}


def cmd_scrape(args):
    if not API_KEY:
        sys.exit("Set MYSCHEME_API_KEY first (browser devtools -> Network tab "
                 "on myscheme.gov.in, x-api-key header of any API call).")
    slugs = [d["id"] for d in _corpus()]
    if args.rules_only:
        keep = _rules_slugs()
        slugs = [s for s in slugs if s in keep]

    for path, url_tpl, label in ((DOCS_PATH, DOCS_URL, "documents"),
                                 (FAQS_PATH, FAQS_URL, "faqs")):
        store = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        todo = [s for s in slugs if s not in store]
        print(f"{label}: {len(todo)} to fetch ({len(store)} cached)", flush=True)
        for i, slug in enumerate(todo, 1):
            try:
                store[slug] = _get(url_tpl.format(slug=urllib.parse.quote(slug)))
            except Exception as e:
                store[slug] = {"status": f"Error: {e!r}"}
            if i % 25 == 0 or i == len(todo):
                path.write_text(json.dumps(store, ensure_ascii=False),
                                encoding="utf-8")
                print(f"  {label}: {i}/{len(todo)}", flush=True)
            time.sleep(RATE_DELAY)
        path.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
    print("done. Now run: python pipeline/scrape_extras.py enrich")


def cmd_enrich(args):
    """Fill documents_required / faqs placeholders in rag_corpus.json."""
    docs = json.loads(DOCS_PATH.read_text(encoding="utf-8")) if DOCS_PATH.exists() else {}
    faqs = json.loads(FAQS_PATH.read_text(encoding="utf-8")) if FAQS_PATH.exists() else {}
    corpus = _corpus()
    n_d = n_f = 0
    for d in corpus:
        entry = docs.get(d["id"])
        if entry and entry.get("status") == "Success":
            en = (entry.get("data") or {}).get("en", {})
            lines = _rich_text(en.get("documents_required", []))
            if lines:
                d["documents_required"] = lines
                n_d += 1
        entry = faqs.get(d["id"])
        if entry and entry.get("status") == "Success":
            en = (entry.get("data") or {}).get("en", {})
            qa = [{"question": f.get("question", ""),
                   "answer": (f.get("answer_md") or " ".join(
                       _rich_text(f.get("answer", []))))[:600]}
                  for f in en.get("faqs", []) if f.get("question")]
            if qa:
                d["faqs"] = qa
                n_f += 1
    with open(DATA / "rag_corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"enriched corpus: {n_d} schemes with documents, {n_f} with FAQs")


def cmd_insights(args):
    """Document-portfolio insight: which documents unlock the most schemes."""
    corpus = [d for d in _corpus() if d.get("documents_required")]
    if not corpus:
        sys.exit("no documents in corpus yet — run scrape + enrich first")
    # normalise document mentions into coarse buckets
    buckets = {
        "aadhaar": "Aadhaar Card", "pan": "PAN Card", "voter": "Voter ID",
        "income certificate": "Income Certificate", "caste": "Caste Certificate",
        "bank": "Bank Account / Passbook", "photo": "Passport-size Photo",
        "residence": "Residence / Domicile Proof", "domicile": "Residence / Domicile Proof",
        "ration": "Ration Card", "disability": "Disability Certificate",
        "age proof": "Age Proof", "birth certificate": "Age Proof",
    }
    counts = Counter()
    for d in corpus:
        seen = set()
        text = " || ".join(d["documents_required"]).lower()
        for key, label in buckets.items():
            if key in text:
                seen.add(label)
        counts.update(seen)
    total = len(corpus)
    print(f"Document portfolio insight ({total} schemes with official checklists):\n")
    for label, n in counts.most_common(10):
        print(f"  {label:32s} needed by {n:4d} schemes ({n/total:5.0%})")
    print("\nTakeaway: a small 'document portfolio' unlocks most schemes — "
          "the agent can tell users exactly which documents to obtain first.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scrape")
    s.add_argument("--rules-only", action="store_true",
                   help="only schemes with eligibility rules (fast, demo-critical)")
    sub.add_parser("enrich")
    sub.add_parser("insights")
    args = ap.parse_args()
    {"scrape": cmd_scrape, "enrich": cmd_enrich, "insights": cmd_insights}[args.cmd](args)


if __name__ == "__main__":
    main()
