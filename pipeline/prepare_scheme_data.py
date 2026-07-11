"""
prepare_scheme_data.py
----------------------
Parses raw myScheme scraped data into:
  1. rag_corpus.json          -> one clean document per scheme (for retrieval + LLM explanation)
  2. scheme_rules_template.csv -> Tier-2 annotation template (manual eligibility fields, blank)

Usage:
    python prepare_scheme_data.py \
        --schemes raw_schemes_min.json \
        --details raw_scheme_details_min.json \
        --outdir  data/

Designed to scale from 5 samples to ~4000 schemes without changes:
  - Tolerates missing keys / missing detail entries (logs and skips gracefully)
  - Never crashes on one bad record; collects errors into a report
"""

import argparse
import csv
import html
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_md(text):
    """Clean markdown text scraped from the portal."""
    if not text:
        return ""
    t = html.unescape(text)              # &amp;amp; -> &  (double-encoded, so run twice)
    t = html.unescape(t)
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)   # literal <br> tags
    t = re.sub(r"<[^>]+>", "", t)                    # any other stray html tags
    t = re.sub(r"[ \t]+", " ", t)                    # collapse spaces
    t = re.sub(r"\n{3,}", "\n\n", t)                 # collapse blank lines
    return t.strip()


def labels(list_of_label_dicts):
    """[{'value':..,'label':'X'}, ...] -> ['X', ...] ; tolerates plain strings."""
    out = []
    for item in list_of_label_dicts or []:
        if isinstance(item, dict):
            out.append(item.get("label") or item.get("value") or "")
        elif isinstance(item, str):
            out.append(item)
    return [x for x in out if x]


def label(maybe_dict):
    """{'value':..,'label':'X'} -> 'X' ; tolerates plain strings / None."""
    if isinstance(maybe_dict, dict):
        return maybe_dict.get("label") or maybe_dict.get("value") or ""
    return maybe_dict or ""


# ---------------------------------------------------------------------------
# Per-scheme parsing
# ---------------------------------------------------------------------------

def parse_scheme(list_entry, detail_entry):
    """
    Merge one scheme-list record with its detail record into a single
    clean RAG document. Either argument may be None.
    """
    lf = (list_entry or {}).get("fields", {})

    # detail payload (may be absent)
    en, slug_from_detail = {}, None
    if detail_entry and detail_entry.get("data"):
        en = detail_entry["data"].get("en", {}) or {}
        slug_from_detail = detail_entry["data"].get("slug")

    bd = en.get("basicDetails", {}) or {}
    sc = en.get("schemeContent", {}) or {}
    ec = en.get("eligibilityCriteria", {}) or {}
    ap = en.get("applicationProcess", []) or []

    slug = lf.get("slug") or slug_from_detail
    name = bd.get("schemeName") or lf.get("schemeName") or ""

    # ---- structured metadata (Tier 1: used for hard filtering) ----
    doc = {
        "id": slug,
        "scheme_name": name,
        "short_title": bd.get("schemeShortTitle") or lf.get("schemeShortTitle") or "",
        "level": label(bd.get("level")) or lf.get("level") or "",           # Central / State
        "states": lf.get("beneficiaryState") or ["All"],
        "scheme_for": bd.get("schemeFor") or lf.get("schemeFor") or "",
        "target_beneficiaries": labels(bd.get("targetBeneficiaries")),
        "categories": labels(bd.get("schemeCategory")) or lf.get("schemeCategory") or [],
        "sub_categories": labels(bd.get("schemeSubCategory")),
        "tags": bd.get("tags") or lf.get("tags") or [],
        "ministry": label(bd.get("nodalMinistryName")) or lf.get("nodalMinistryName") or "",
        "department": label(bd.get("nodalDepartmentName")),
        "dbt_scheme": bd.get("dbtScheme", None),
        "benefit_type": label(sc.get("benefitTypes")),
        "open_date": bd.get("schemeOpenDate"),
        "close_date": bd.get("schemeCloseDate") or lf.get("schemeCloseDate"),

        # ---- text sections (Tier: RAG / LLM explanation) ----
        "brief_description": clean_md(sc.get("briefDescription") or lf.get("briefDescription")),
        "detailed_description": clean_md(sc.get("detailedDescription_md")),
        "benefits": clean_md(sc.get("benefits_md")),
        "eligibility_text": clean_md(ec.get("eligibilityDescription_md")),
        "exclusions": clean_md(sc.get("exclusions_md")),

        # ---- application process ----
        "application": [
            {
                "mode": p.get("mode"),
                "url": (p.get("url") or "").strip() or None,
                "process": clean_md(p.get("process_md")),
            }
            for p in ap
        ],
        "references": [
            {"title": r.get("title"), "url": (r.get("url") or "").strip()}
            for r in sc.get("references", []) or []
        ],

        # ---- placeholders for future scraping ----
        "documents_required": [],   # fill when you scrape this endpoint
        "faqs": [],                 # fill when you scrape this endpoint
    }

    # single concatenated field for keyword retrieval (search over this)
    doc["search_text"] = " ".join(filter(None, [
        doc["scheme_name"],
        doc["short_title"],
        " ".join(doc["tags"]),
        " ".join(doc["categories"]),
        " ".join(doc["sub_categories"]),
        doc["brief_description"],
        doc["eligibility_text"],
    ])).lower()

    return doc


# ---------------------------------------------------------------------------
# Rules template (Tier 2 - manual annotation)
# ---------------------------------------------------------------------------

RULES_COLUMNS = [
    "id", "scheme_name", "level", "states",
    # numeric criteria (leave blank = not applicable)
    "age_min", "age_max", "income_max_annual",
    # categorical criteria (leave blank = any)
    "gender",            # male / female / any
    "category",          # SC / ST / OBC / EWS / any  (comma-sep if multiple)
    "occupation",        # farmer / student / faculty / entrepreneur / artist / any
    "marital_status",    # widow / any ...
    "requires_bank_account",   # yes / blank
    "land_owner",              # yes / no / blank
    "other_conditions",        # free text for anything not capturable above
    "documents_required",      # comma-sep, manual for now
    # keep source text beside the blanks so the annotator never has to
    # open the portal while filling this in
    "eligibility_text_source",
]


def write_rules_template(docs, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RULES_COLUMNS)
        w.writeheader()
        for d in docs:
            w.writerow({
                "id": d["id"],
                "scheme_name": d["scheme_name"],
                "level": d["level"],
                "states": "|".join(d["states"]),
                "eligibility_text_source": d["eligibility_text"][:1500],
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--schemes", required=True, help="raw scheme list json")
    p.add_argument("--details", required=True, help="raw scheme details json (keyed by slug)")
    p.add_argument("--outdir", default="data", help="output directory")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.schemes, encoding="utf-8") as f:
        scheme_list = json.load(f)
    with open(args.details, encoding="utf-8") as f:
        details = json.load(f)

    # index list entries by slug
    list_by_slug = {}
    for entry in scheme_list:
        slug = entry.get("fields", {}).get("slug")
        if slug:
            list_by_slug[slug] = entry

    all_slugs = sorted(set(list_by_slug) | set(details))

    docs, errors = [], []
    for slug in all_slugs:
        try:
            detail_entry = details.get(slug)
            if detail_entry and detail_entry.get("status") != "Success":
                errors.append({"slug": slug, "error": "detail status not Success"})
                detail_entry = None
            doc = parse_scheme(list_by_slug.get(slug), detail_entry)
            if not doc["scheme_name"]:
                errors.append({"slug": slug, "error": "no scheme name resolvable"})
                continue
            docs.append(doc)
        except Exception as e:  # never let one bad record kill a 4000-scheme run
            errors.append({"slug": slug, "error": repr(e)})

    # outputs
    corpus_path = outdir / "rag_corpus.json"
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    rules_path = outdir / "scheme_rules_template.csv"
    write_rules_template(docs, rules_path)

    report_path = outdir / "parse_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_slugs": len(all_slugs),
            "parsed_ok": len(docs),
            "errors": errors,
            "missing_detail": [s for s in list_by_slug if s not in details],
            "missing_list_entry": [s for s in details if s not in list_by_slug],
        }, f, indent=2)

    print(f"parsed {len(docs)}/{len(all_slugs)} schemes")
    print(f"  -> {corpus_path}")
    print(f"  -> {rules_path}  (fill the blank columns manually for demo schemes)")
    print(f"  -> {report_path}")
    if errors:
        print(f"  !! {len(errors)} errors, see parse_report.json")


if __name__ == "__main__":
    main()