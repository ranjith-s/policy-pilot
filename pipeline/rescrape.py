"""
rescrape.py — Re-scrape the myScheme portal (list + per-scheme detail APIs).

The portal exposes two public APIs (the same ones the myscheme.gov.in
frontend calls; the x-api-key is visible in the site's own network traffic):

  1. search list : GET {API_BASE}/search/v5/schemes?...  (paginated)
  2. detail      : GET {API_BASE}/schemes/v5/public/schemes?slug=<slug>

Outputs the same two files the original scrape produced:
  raw_schemes.json        — list of {"id", "fields": {...}} entries
  raw_scheme_details.json — {slug: {"status": "Success", "data": {...}}}

Etiquette: single-threaded, RATE_DELAY seconds between requests, resumable
(details fetch skips slugs already present). This is a public government
API — never parallelise or hammer it.

Usage:
    python pipeline/rescrape.py --outdir data_raw [--limit 20]   # smoke test
    python pipeline/rescrape.py --outdir data_raw                # full scrape
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API_BASE = os.environ.get("MYSCHEME_API_BASE", "https://api.myscheme.gov.in")
# The portal frontend sends an x-api-key with every request. Set it via env:
#   export MYSCHEME_API_KEY=...   (find it in browser devtools -> Network tab
#   on myscheme.gov.in, request headers of any /search or /schemes call)
API_KEY = os.environ.get("MYSCHEME_API_KEY", "")
RATE_DELAY = 1.0   # seconds between requests — be gentle, it's a public service


def _get(url):
    req = urllib.request.Request(url, headers={
        "x-api-key": API_KEY,
        "Accept": "application/json",
        "User-Agent": "scheme-agent-refresh/1.0 (student capstone)",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def scrape_list(limit=0):
    """Page through the search API; returns raw_schemes-style list."""
    out, page_size, offset = [], 100, 0
    while True:
        q = urllib.parse.quote(json.dumps([]))
        url = (f"{API_BASE}/search/v5/schemes?lang=en&q={q}"
               f"&keyword=&sort=&from={offset}&size={page_size}")
        data = _get(url)
        items = data.get("data", {}).get("hits", {}).get("items", [])
        if not items:
            break
        out.extend(items)
        print(f"  list: {len(out)} schemes", flush=True)
        offset += page_size
        if limit and len(out) >= limit:
            out = out[:limit]
            break
        time.sleep(RATE_DELAY)
    return out


def scrape_details(slugs, existing, limit=0):
    """Fetch per-scheme detail; skips slugs already in `existing`."""
    todo = [s for s in slugs if s not in existing]
    if limit:
        todo = todo[:limit]
    print(f"  details: {len(todo)} to fetch ({len(existing)} cached)", flush=True)
    for i, slug in enumerate(todo, 1):
        try:
            data = _get(f"{API_BASE}/schemes/v5/public/schemes?"
                        f"slug={urllib.parse.quote(slug)}&lang=en")
            existing[slug] = data
        except Exception as e:
            existing[slug] = {"status": f"Error: {e!r}"}
        if i % 25 == 0 or i == len(todo):
            print(f"  details: {i}/{len(todo)}", flush=True)
        time.sleep(RATE_DELAY)
    return existing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data_raw")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: cap schemes")
    args = ap.parse_args()

    if not API_KEY:
        sys.exit("Set MYSCHEME_API_KEY first (see comment at top of this file).")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    list_path = outdir / "raw_schemes.json"
    det_path = outdir / "raw_scheme_details.json"

    print("scraping scheme list ...", flush=True)
    schemes = scrape_list(limit=args.limit)
    list_path.write_text(json.dumps(schemes, ensure_ascii=False), encoding="utf-8")

    slugs = [s.get("fields", {}).get("slug") for s in schemes]
    slugs = [s for s in slugs if s]
    existing = {}
    if det_path.exists():
        existing = json.loads(det_path.read_text(encoding="utf-8"))
    details = scrape_details(slugs, existing, limit=args.limit)
    det_path.write_text(json.dumps(details, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {list_path} ({len(schemes)}) and {det_path} ({len(details)})")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
