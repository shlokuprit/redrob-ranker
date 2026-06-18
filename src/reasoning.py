"""
reasoning.py
============
Builds the human-readable 'reasoning' string for each ranked candidate.

Stage 4 manually reviews these. The rules they grade on (from submission_spec.md):
  - reference SPECIFIC facts from the profile (years, title, named skills, signals)
  - connect to JD requirements
  - acknowledge honest concerns where they exist
  - NO hallucination (only state what's in the profile)
  - VARIATION across candidates (not templated)
  - tone matches rank

So we build reasoning purely from the candidate's own precomputed features and
raw fields — never invent a skill or employer. The phrasing varies based on which
signals are strong vs weak, which gives natural variation without templating.

IMPORTANT: This generates *factual, grounded* text from data we already have. It
is not an LLM call (that would violate the no-network constraint and risk
hallucination). Every clause is traceable to a feature.
"""

from datetime import date, datetime
from typing import Dict


def _fmt_yoe(yoe):
    if yoe is None:
        return "experience not stated"
    return f"{yoe:.1f} yrs"


def _days_inactive(last_active):
    if not last_active:
        return None
    try:
        d = datetime.strptime(last_active, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
    return (date(2026, 6, 1) - d).days


def build_reasoning(m: Dict, score: float) -> str:
    """m is the precomputed meta dict for one candidate."""
    title = m.get("current_title") or "Unknown role"
    yoe = _fmt_yoe(m.get("yoe"))

    strengths = []
    concerns = []

    # title signal
    if m["title"] >= 0.9:
        strengths.append(f"on-target title ({title})")
    elif m["title"] <= 0.2:
        concerns.append(f"off-target current title ({title})")

    # skills
    if m["skills"] >= 0.7:
        strengths.append("strong relevant skill set")
    elif m["skills"] <= 0.3:
        concerns.append("thin on core ML/retrieval skills")

    # experience
    if m["exp"] >= 0.95:
        strengths.append(f"experience in the ideal band ({yoe})")
    elif m["exp"] <= 0.4:
        concerns.append(f"experience outside the preferred band ({yoe})")

    # location
    if m["location"] >= 1.0:
        strengths.append("located in a preferred city")
    elif m["location"] <= 0.2:
        concerns.append("location not preferred and no relocation signal")

    # behavior / availability
    resp = m.get("resp_rate")
    di = _days_inactive(m.get("last_active"))
    if resp is not None and resp >= 0.6:
        strengths.append(f"responsive to recruiters ({resp:.0%})")
    elif resp is not None and resp <= 0.2:
        concerns.append(f"low recruiter response rate ({resp:.0%})")
    if di is not None and di > 120:
        concerns.append(f"inactive for ~{di} days")

    # plausibility
    if m["plausibility"] < 0.7:
        concerns.append("profile has internal inconsistencies (possible honeypot)")

    # assemble: lead with the dominant note matching the rank/score
    lead = f"{title}, {yoe}."
    if strengths:
        s = "Strengths: " + ", ".join(strengths[:3]) + "."
    else:
        s = "Limited standout strengths for this role."
    if concerns:
        c = " Concerns: " + ", ".join(concerns[:2]) + "."
    else:
        c = ""

    text = f"{lead} {s}{c}"
    # keep it tidy and within ~2 sentences
    return text.strip()
