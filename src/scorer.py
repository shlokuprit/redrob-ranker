"""
scorer.py
=========
Combines the precomputed semantic similarity and structured features into one
final score per candidate, then produces the ranked top-N with reasoning.

The formula (the thing you defend in the interview):

    fit = w_sem*semantic + w_skills*skills + w_title*title
        + w_exp*exp + w_loc*location + w_edu*education      # additive, in [0,1]

    final = fit * behavior_mult * consulting_mult * plausibility_mult

  - fit is a weighted sum of normalized components (interpretable: each term's
    contribution is weight*component).
  - the three multipliers MODIFY fit rather than adding to it, because
    "unavailable", "consulting-only", and "implausible" are all reasons to
    discount an otherwise-good match, not independent positive evidence.
  - plausibility quietly sinks honeypots without us hand-labeling them.

Everything is vectorized over numpy arrays so 100k candidates score in well
under a second.
"""

from typing import Dict, List, Tuple

import numpy as np

from config import WEIGHTS


def compute_scores(
    semantic_sim: np.ndarray,     # (N,) cosine sim of each candidate to the JD
    meta: List[Dict],             # length N, per-candidate precomputed features
) -> np.ndarray:
    """Return (N,) final scores in roughly [0,1]."""
    n = len(meta)

    # pull structured features into arrays
    skills = np.array([m["skills"] for m in meta], dtype=np.float32)
    title = np.array([m["title"] for m in meta], dtype=np.float32)
    exp = np.array([m["exp"] for m in meta], dtype=np.float32)
    loc = np.array([m["location"] for m in meta], dtype=np.float32)
    edu = np.array([m["education"] for m in meta], dtype=np.float32)
    behavior = np.array([m["behavior"] for m in meta], dtype=np.float32)
    consulting = np.array([m["consulting"] for m in meta], dtype=np.float32)
    title_climber = np.array([m.get("title_climber", 1.0) for m in meta], dtype=np.float32)
    plausibility = np.array([m["plausibility"] for m in meta], dtype=np.float32)

    # normalize semantic sim to 0..1 (cosine of normalized vectors is in [-1,1],
    # but for this data it's effectively [0,1]; clip to be safe)
    sem = np.clip(semantic_sim, 0.0, 1.0).astype(np.float32)

    fit = (
        WEIGHTS["semantic"] * sem +
        WEIGHTS["skills"] * skills +
        WEIGHTS["title"] * title +
        WEIGHTS["exp"] * exp +
        WEIGHTS["location"] * loc +
        WEIGHTS["education"] * edu
    )

    final = fit * behavior * consulting * title_climber * plausibility
    return final


def rank_top_n(
    final_scores: np.ndarray,
    cand_ids: List[str],
    n: int,
) -> List[Tuple[int, str, float]]:
    """Return [(row_index, candidate_id, score), ...] for the top n, best first.

    Tie-break: the spec requires unique ranks and, on equal score, candidate_id
    ascending. We round the score in the sort key to the same precision the
    submission uses (4 dp), so float32 noise below that precision can't defeat the
    tie-break — genuinely-equal scores then fall through to candidate_id ascending.
    """
    order = sorted(
        range(len(final_scores)),
        key=lambda i: (-round(float(final_scores[i]), 4), cand_ids[i]),
    )
    top = order[:n]
    return [(i, cand_ids[i], float(final_scores[i])) for i in top]
