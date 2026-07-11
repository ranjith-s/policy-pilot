"""
engine.py — Deterministic eligibility engine (NO LLM here).

This is the source of truth for eligibility. The agent may only report
what this module returns; it can never decide eligibility itself.
"""

import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# profile fields the engine understands
KNOWN_FIELDS = [
    "age", "annual_income", "gender", "category", "occupation",
    "state", "marital_status", "has_bank_account", "land_owner",
]

# human-friendly phrasings the agent can use when asking for a field
FIELD_QUESTIONS = {
    "age": "What is your age?",
    "annual_income": "What is your total annual income (in rupees)?",
    "gender": "What is your gender?",
    "category": "Which category do you belong to (SC / ST / OBC / EWS / General)?",
    "occupation": "What is your occupation (e.g. farmer, student, artist, entrepreneur, faculty)?",
    "state": "Which state do you live in?",
    "marital_status": "What is your marital status (single / married / widow)?",
    "has_bank_account": "Do you have an active bank account? (yes/no)",
    "land_owner": "Do you own agricultural land? (yes/no)",
}


# columns that count as an actual annotation; a row where ALL of these are
# blank is an unannotated template row and must never produce a verdict
ANNOTATION_COLUMNS = [
    "age_min", "age_max", "income_max_annual", "gender", "category",
    "occupation", "marital_status", "requires_bank_account", "land_owner",
    "other_conditions", "documents_required",
]


def _is_annotated(row):
    return any((row.get(c) or "").strip() for c in ANNOTATION_COLUMNS)


def _load_rules(rules_csv=None):
    path = Path(rules_csv) if rules_csv else DATA_DIR / "scheme_rules.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if _is_annotated(r)]


def _num(value):
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _check_one(rule, profile):
    """Evaluate one scheme rule row against a profile.

    Returns (status, reasons, missing_fields)
      status: eligible | partial | not_eligible
    """
    reasons, missing = [], []
    failed = False

    def need(field):
        """Mark a profile field as required-but-missing."""
        if field not in missing:
            missing.append(field)

    # ---- age ----
    age_min, age_max = _num(rule.get("age_min")), _num(rule.get("age_max"))
    if age_min is not None or age_max is not None:
        age = _num(profile.get("age"))
        if age is None:
            need("age")
        else:
            if age_min is not None and age < age_min:
                failed = True
                reasons.append(f"Minimum age is {int(age_min)}, you are {int(age)}")
            if age_max is not None and age > age_max:
                failed = True
                reasons.append(f"Maximum age is {int(age_max)}, you are {int(age)}")
            if not failed and (age_min is not None or age_max is not None):
                reasons.append("Age requirement met")

    # ---- income ----
    income_max = _num(rule.get("income_max_annual"))
    if income_max is not None:
        income = _num(profile.get("annual_income"))
        if income is None:
            need("annual_income")
        elif income > income_max:
            failed = True
            reasons.append(
                f"Annual income must be under ₹{int(income_max):,}, yours is ₹{int(income):,}"
            )
        else:
            reasons.append(f"Income within ₹{int(income_max):,} limit")

    # ---- gender ----
    rule_gender = (rule.get("gender") or "").strip().lower()
    if rule_gender:
        gender = (profile.get("gender") or "").strip().lower()
        if not gender:
            need("gender")
        elif gender != rule_gender:
            failed = True
            reasons.append(f"Scheme is for {rule_gender} applicants")
        else:
            reasons.append("Gender requirement met")

    # ---- category (supports special token female_any_category e.g. Stand-Up India) ----
    rule_cat = (rule.get("category") or "").strip()
    if rule_cat:
        allowed = [c.strip().lower() for c in rule_cat.split(",") if c.strip()]
        female_ok = "female_any_category" in allowed
        cat = (profile.get("category") or "").strip().lower()
        gender = (profile.get("gender") or "").strip().lower()

        if female_ok and gender == "female":
            reasons.append("Eligible as a woman applicant (any category)")
        else:
            plain_allowed = [a for a in allowed if a != "female_any_category"]
            if not cat and not (female_ok and not gender):
                need("category")
                if female_ok and not gender:
                    need("gender")
            elif cat and plain_allowed and cat not in plain_allowed:
                if female_ok and not gender:
                    need("gender")  # could still qualify as female
                else:
                    failed = True
                    reasons.append(
                        f"Scheme is for {'/'.join(a.upper() for a in plain_allowed)}"
                        + (" or women of any category" if female_ok else "")
                    )
            elif cat and cat in plain_allowed:
                reasons.append(f"Category ({cat.upper()}) requirement met")

    # ---- occupation ----
    rule_occ = (rule.get("occupation") or "").strip().lower()
    if rule_occ:
        occ = (profile.get("occupation") or "").strip().lower()
        if not occ:
            need("occupation")
        elif occ != rule_occ:
            failed = True
            reasons.append(f"Scheme is for {rule_occ}s")
        else:
            reasons.append(f"Occupation ({occ}) matches")

    # ---- state ----
    rule_states = [s.strip().lower() for s in (rule.get("states") or "All").split("|")]
    if rule_states and "all" not in rule_states:
        state = (profile.get("state") or "").strip().lower()
        if not state:
            need("state")
        elif state not in rule_states:
            failed = True
            reasons.append(f"Scheme only for residents of {rule.get('states')}")
        else:
            reasons.append("State requirement met")

    # ---- marital status ----
    rule_ms = (rule.get("marital_status") or "").strip().lower()
    if rule_ms:
        ms = (profile.get("marital_status") or "").strip().lower()
        if not ms:
            need("marital_status")
        elif ms != rule_ms:
            failed = True
            reasons.append(f"Scheme is for {rule_ms}s")
        else:
            reasons.append("Marital status requirement met")

    # ---- bank account ----
    if (rule.get("requires_bank_account") or "").strip().lower() == "yes":
        hba = profile.get("has_bank_account")
        if hba is None or str(hba).strip() == "":
            need("has_bank_account")
        elif str(hba).strip().lower() in ("no", "false", "0"):
            failed = True
            reasons.append("An active bank account is required")
        else:
            reasons.append("Bank account requirement met")

    if failed:
        return "not_eligible", reasons, []
    if missing:
        return "partial", reasons, missing
    return "eligible", reasons, []


def check_eligibility(profile, rules_csv=None):
    """Check a user profile against every scheme rule.

    Returns list of:
      {scheme_id, scheme_name, status, reasons, missing_fields,
       other_conditions, documents_required}
    """
    results = []
    for rule in _load_rules(rules_csv):
        status, reasons, missing = _check_one(rule, profile)
        results.append({
            "scheme_id": rule["id"],
            "scheme_name": rule["scheme_name"],
            "status": status,
            "reasons": reasons,
            "missing_fields": missing,
            "other_conditions": rule.get("other_conditions", ""),
            "documents_required": [
                d.strip() for d in (rule.get("documents_required") or "").split(",") if d.strip()
            ],
        })
    return results


def get_next_question(profile, results):
    """Pick the single most useful missing field to ask about next.

    Deterministic: the field missing across the most 'partial' schemes wins.
    Returns {"field", "question", "blocking_schemes"} or None.
    """
    counts = {}
    for r in results:
        if r["status"] == "partial":
            for f in r["missing_fields"]:
                counts.setdefault(f, []).append(r["scheme_name"])
    if not counts:
        return None
    field = max(counts, key=lambda f: len(counts[f]))
    return {
        "field": field,
        "question": FIELD_QUESTIONS.get(field, f"Please provide your {field}."),
        "blocking_schemes": counts[field],
    }
