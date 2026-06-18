"""
features.py
===========
Turns a raw candidate dict into structured numeric features that the scorer can
combine with the semantic similarity. This is the "hybrid" half of hybrid
scoring — the part that catches what pure embeddings miss.

Each function returns a value in a documented range so the scorer can combine
them on a common scale. Every feature here is something you can explain in one
sentence to a human judge.

The honeypot detector lives here too. The spec says honeypots have *subtly
impossible* profiles; ranking them in your top 10 gets you disqualified. We
don't special-case them in scoring — instead we compute a plausibility penalty
that naturally pushes impossible profiles down, which is the behavior the
organizers say a good system should show "naturally."
"""

from datetime import date, datetime
from typing import Dict, List

from config import (
    BEHAVIOR_MULTIPLIER_MAX,
    BEHAVIOR_MULTIPLIER_MIN,
    CONSULTING_FIRMS,
    CORE_SKILLS,
    EXP_IDEAL_MAX,
    EXP_IDEAL_MIN,
    EXP_SWEET_MAX,
    EXP_SWEET_MIN,
    NEGATIVE_TITLE_KEYWORDS,
    NICE_TO_HAVE_SKILLS,
    POSITIVE_TITLE_KEYWORDS,
    PREFERRED_LOCATIONS,
    REQUIRED_SKILLS,
)

# A fixed "today" so scoring is deterministic and reproducible (Stage 3 re-runs
# your code; a moving datetime.now() would make recency scores drift run-to-run).
REFERENCE_DATE = date(2026, 6, 1)


def _lc(s) -> str:
    return (s or "").lower()


def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------
def skills_score(c: Dict) -> float:
    """0..1 weighted overlap of candidate skills with what the JD needs.

    We trust a skill more when it's backed by endorsements and real duration —
    this is the lazy-keyword-stuffer defense. A skill listed 'expert' with 0
    months of use earns almost nothing.
    """
    skills = c.get("skills", [])
    if not skills:
        return 0.0

    core = set(CORE_SKILLS)
    nice = set(NICE_TO_HAVE_SKILLS)
    req = set(REQUIRED_SKILLS)

    total = 0.0
    for s in skills:
        name = _lc(s.get("name"))
        if not name:
            continue
        # base weight by importance bucket
        if any(k in name or name in k for k in req):
            base = 1.0
        elif any(k in name or name in k for k in core):
            base = 0.8
        elif any(k in name or name in k for k in nice):
            base = 0.3
        else:
            continue  # irrelevant skill contributes nothing

        # trust multiplier: endorsements + duration temper raw self-claims
        dur = s.get("duration_months", 0) or 0
        endo = s.get("endorsements", 0) or 0
        trust = 0.4 + 0.4 * min(dur / 24.0, 1.0) + 0.2 * min(endo / 20.0, 1.0)
        total += base * trust

    # normalize: ~5 strong relevant skills should approach 1.0
    return min(total / 4.0, 1.0)


def has_required_skills(c: Dict) -> bool:
    names = " ".join(_lc(s.get("name")) for s in c.get("skills", []))
    summary = _lc(c.get("profile", {}).get("summary"))
    blob = names + " " + summary
    return all(any(part in blob for part in [r]) for r in REQUIRED_SKILLS)


# ---------------------------------------------------------------------------
# Title alignment  (the decisive anti-stuffing signal)
# ---------------------------------------------------------------------------
def title_score(c: Dict) -> float:
    """-? mapped to 0..1. Looks at current title AND recent titles.

    The JD is explicit: a Marketing Manager with every AI keyword is NOT a fit.
    Title is how we encode 'is this person actually doing this kind of work.'
    """
    p = c.get("profile", {})
    current = _lc(p.get("current_title"))
    recent_titles = [current] + [
        _lc(r.get("title")) for r in c.get("career_history", [])[:2]
    ]

    pos = any(
        any(k in t for k in POSITIVE_TITLE_KEYWORDS) for t in recent_titles if t
    )
    neg = any(k in current for k in NEGATIVE_TITLE_KEYWORDS)

    if pos and not neg:
        return 1.0
    if pos and neg:
        return 0.6           # mixed: maybe transitioning into ML
    if not pos and not neg:
        return 0.4           # neutral / ambiguous title
    return 0.1               # clearly off-target current title


# ---------------------------------------------------------------------------
# Experience band
# ---------------------------------------------------------------------------
def experience_score(c: Dict) -> float:
    """0..1, peaked on the JD's sweet spot (6-8y), still generous across 5-9y,
    and degrading outside the band rather than hard-cutting (the JD says it'll
    consider out-of-band candidates if other signals are strong)."""
    yoe = c.get("profile", {}).get("years_of_experience")
    if yoe is None:
        return 0.3
    if EXP_SWEET_MIN <= yoe <= EXP_SWEET_MAX:
        return 1.0
    if EXP_IDEAL_MIN <= yoe <= EXP_IDEAL_MAX:
        return 0.85
    # linear falloff outside the ideal band
    if yoe < EXP_IDEAL_MIN:
        return max(0.0, 0.85 - (EXP_IDEAL_MIN - yoe) * 0.20)
    return max(0.0, 0.85 - (yoe - EXP_IDEAL_MAX) * 0.10)


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------
def location_score(c: Dict) -> float:
    p = c.get("profile", {})
    loc = _lc(p.get("location")) + " " + _lc(p.get("country"))
    if any(city in loc for city in PREFERRED_LOCATIONS):
        return 1.0
    # willing to relocate softens a non-preferred location
    if c.get("redrob_signals", {}).get("willing_to_relocate"):
        return 0.6
    if "india" in loc:
        return 0.5
    return 0.2


# ---------------------------------------------------------------------------
# Education (mild signal)
# ---------------------------------------------------------------------------
def education_score(c: Dict) -> float:
    edu = c.get("education", [])
    if not edu:
        return 0.4
    tier_map = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.6,
                "tier_4": 0.4, "unknown": 0.5}
    best = max((tier_map.get(e.get("tier", "unknown"), 0.5) for e in edu),
              default=0.5)
    return best


# ---------------------------------------------------------------------------
# Consulting-only penalty (soft)
# ---------------------------------------------------------------------------
def consulting_only_penalty(c: Dict) -> float:
    """Returns a multiplier <=1. The JD dislikes pure-services careers but is OK
    if there's any product-company experience. So we only penalize when EVERY
    role is at a consulting firm."""
    history = c.get("career_history", [])
    if not history:
        return 1.0
    companies = [_lc(r.get("company")) for r in history]
    all_consulting = all(
        any(firm in comp for firm in CONSULTING_FIRMS) for comp in companies
    )
    return 0.85 if all_consulting else 1.0


# ---------------------------------------------------------------------------
# Title-chaser detection (soft)
# ---------------------------------------------------------------------------
# The JD names a specific disqualifier: someone "optimizing for Senior -> Staff
# -> Principal titles by switching companies every 1.5 years." The key is that
# it's CLIMBING + HOPPING together, not short tenure alone. A lateral mover
# (ML -> Search -> NLP, all same level) is NOT a title-chaser and is often a
# *strength* for a retrieval role. So we detect the actual pattern: short average
# tenure AND a genuinely ascending seniority ladder. We verified against the real
# top-100 that this spares lateral movers and erratic histories, firing only on a
# true climb. Penalty is SOFT (0.93x) because title-climbing is an inference, not
# a certainty like an impossible profile — we calibrate penalty to confidence.

# Seniority tiers, low -> high. We take the highest tier word present in a title.
_SENIORITY_TIERS = [
    (["intern", "trainee"], 0),
    (["junior", "jr", "associate"], 1),
    (["senior", "sr"], 3),
    (["lead", "staff"], 4),
    (["principal", "head", "director", "vp", "chief"], 5),
]


def _seniority_rank(title: str) -> int:
    """Map a title to a seniority level. Base individual-contributor = 2; modifiers
    push it up or down. Returns the HIGHEST tier word found in the title."""
    t = _lc(title)
    best = 2  # default: a plain "Engineer"/"Scientist" with no seniority word
    for words, rank in _SENIORITY_TIERS:
        if any(w in t for w in words):
            best = max(best, rank)
    return best


def title_climber_penalty(c: Dict) -> float:
    """Returns a multiplier <=1. 1.0 = not a title-chaser. 0.93 = matches the JD's
    climb-while-hopping pattern.

    Both conditions must hold:
      1. 3+ jobs with short average tenure (< 18 months) -> frequent switching
      2. seniority trends genuinely UPWARD across those jobs (ends higher than it
         started, and more upward steps than downward) -> a real climb, not bounce
    """
    history = c.get("career_history", [])
    durations = [r.get("duration_months", 0) or 0 for r in history
                 if r.get("duration_months")]
    if len(durations) < 3:
        return 1.0
    if sum(durations) / len(durations) >= 18:
        return 1.0   # not a frequent switcher

    # build seniority sequence oldest -> newest
    ranks = [_seniority_rank(r.get("title", ""))
             for r in reversed(history) if r.get("title")]
    if len(ranks) < 3:
        return 1.0
    ups = sum(1 for a, b in zip(ranks, ranks[1:]) if b > a)
    downs = sum(1 for a, b in zip(ranks, ranks[1:]) if b < a)
    net_climb = ranks[-1] - ranks[0]

    # A genuine title-chaser shows a SUSTAINED climb: multiple upward steps across
    # the hops (Junior->Senior->Lead->Principal), not a single early promotion
    # (Junior->Senior then steady), which is healthy growth. Requiring >=2 upward
    # steps is what separates the JD's disqualifying pattern from normal progress.
    is_climber = net_climb >= 2 and ups >= 2 and ups > downs
    return 0.93 if is_climber else 1.0


# ---------------------------------------------------------------------------
# Behavioral multiplier
# ---------------------------------------------------------------------------
def behavior_multiplier(c: Dict) -> float:
    """Collapse the 23 redrob signals into ONE bounded multiplier.

    Philosophy (from the signals doc): availability/responsiveness modifies a
    candidate's practical value. Great-on-paper but unreachable => down-weight.
    We bound it to [0.70, 1.10] so it tilts the ranking without overriding fit.
    """
    s = c.get("redrob_signals", {})

    # recency of activity
    last = _parse_date(s.get("last_active_date"))
    if last:
        days_inactive = (REFERENCE_DATE - last).days
        recency = max(0.0, 1.0 - days_inactive / 180.0)   # 0 at ~6mo idle
    else:
        recency = 0.3

    resp = s.get("recruiter_response_rate", 0.0) or 0.0     # 0..1
    open_flag = 1.0 if s.get("open_to_work_flag") else 0.5
    completeness = (s.get("profile_completeness_score", 0) or 0) / 100.0
    interview = s.get("interview_completion_rate", 0.0) or 0.0
    saved = min((s.get("saved_by_recruiters_30d", 0) or 0) / 10.0, 1.0)

    # weighted blend of availability signals, all in 0..1
    avail = (
        0.30 * recency +
        0.25 * resp +
        0.15 * open_flag +
        0.10 * completeness +
        0.10 * interview +
        0.10 * saved
    )

    # map avail (0..1) onto [MIN, MAX]
    return BEHAVIOR_MULTIPLIER_MIN + avail * (
        BEHAVIOR_MULTIPLIER_MAX - BEHAVIOR_MULTIPLIER_MIN
    )


# ---------------------------------------------------------------------------
# Honeypot / plausibility detection
# ---------------------------------------------------------------------------
def implausibility_penalty(c: Dict) -> float:
    """Returns a multiplier in (0,1]. 1.0 = fully plausible. Lower = something
    is internally inconsistent, the hallmark of a honeypot.

    Checks (each subtracts from a starting 1.0):
      - 'expert' skill with 0 months of use
      - tenure at a company exceeding the company's plausible age (we proxy via
        sum of role durations vs years_of_experience)
      - years_of_experience wildly inconsistent with career history total
      - skill claims with impossible duration (> years_of_experience*12 + slack)
    These mirror the examples in submission_spec.md Section 7.
    """
    penalty = 0.0
    p = c.get("profile", {})
    yoe = p.get("years_of_experience") or 0
    yoe_months = yoe * 12

    # 1) expert-but-unused skills
    for s in c.get("skills", []):
        if s.get("proficiency") == "expert" and (s.get("duration_months", 0) or 0) == 0:
            penalty += 0.25
            break

    # 2) any single skill claims more months than the whole career allows
    for s in c.get("skills", []):
        dur = s.get("duration_months", 0) or 0
        if yoe_months and dur > yoe_months + 24:   # 2yr slack for overlap noise
            penalty += 0.25
            break

    # 3) career duration vs stated experience grossly inconsistent
    hist_months = sum(r.get("duration_months", 0) or 0 for r in c.get("career_history", []))
    # roles can overlap, so hist can exceed yoe somewhat; flag only large gaps
    if yoe_months and hist_months > yoe_months * 1.8 + 24:
        penalty += 0.20

    # 4) a single role longer than the entire stated career
    for r in c.get("career_history", []):
        if yoe_months and (r.get("duration_months", 0) or 0) > yoe_months + 24:
            penalty += 0.20
            break

    return max(0.15, 1.0 - penalty)   # floor so we never hard-zero by heuristic
