"""
data_io.py
==========
Everything about *reading* candidates and turning each one into a single text
string for embedding. Two responsibilities only:

  1. stream_candidates()  -> yields raw dicts from the .jsonl (memory-safe)
  2. candidate_to_text()  -> builds the text we embed for semantic matching

Why a dedicated "to_text" function? The embedding model only sees text. What you
put in (and the order) is a real design choice: we lead with the signals that
matter most for *this* JD (headline, current title, summary, recent role
descriptions) so the meaning-vector is dominated by what actually predicts fit,
not by, say, a long list of unrelated skills. That's the difference between
"semantic understanding" and "keyword soup."
"""

import json
from typing import Dict, Iterator, List


def stream_candidates(path) -> Iterator[Dict]:
    """Yield one candidate dict per line. Streaming keeps RAM flat even on 100k
    records (we never hold the whole file as parsed objects unless we choose to)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_all(path) -> List[Dict]:
    """Load everything into a list. Fine for 100k records on 16GB, and simpler
    for the ranking step where we need random access. ~500MB raw -> a few GB
    parsed; still within budget."""
    return list(stream_candidates(path))


def candidate_to_text(c: Dict) -> str:
    """Serialize a candidate into the text we embed.

    Order matters: most-predictive fields first. We deliberately include the
    free-text role *descriptions* because that's where a Tier-5 candidate ('built
    a recommendation system at a product company') reveals real fit without ever
    using buzzwords like 'RAG'. We keep skills but don't let them dominate."""
    p = c.get("profile", {})
    parts: List[str] = []

    # 1) The strongest single-line signals.
    if p.get("current_title"):
        parts.append(f"Current role: {p['current_title']}.")
    if p.get("headline"):
        parts.append(p["headline"])
    if p.get("years_of_experience") is not None:
        parts.append(f"{p['years_of_experience']} years of experience.")

    # 2) The summary — usually the richest semantic content.
    if p.get("summary"):
        parts.append(p["summary"])

    # 3) Recent career history descriptions (cap at the 3 most recent roles so a
    #    10-job history doesn't drown out the signal). These descriptions are
    #    where genuine systems-building shows up.
    history = c.get("career_history", [])[:3]
    for role in history:
        title = role.get("title", "")
        company = role.get("company", "")
        desc = role.get("description", "")
        line = f"{title} at {company}. {desc}".strip()
        if line:
            parts.append(line)

    # 4) Skills, but as a compact phrase, not the centerpiece.
    skills = c.get("skills", [])
    if skills:
        skill_names = ", ".join(s.get("name", "") for s in skills if s.get("name"))
        if skill_names:
            parts.append(f"Skills: {skill_names}.")

    return "\n".join(parts)
