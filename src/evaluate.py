"""
evaluate.py
===========
Local evaluation harness. The real ground truth is hidden, so you CANNOT compute
your true competition score. What you CAN do — and should, since there's no live
leaderboard — is:

  1. Build a small *self-labeled* validation set: hand-pick ~30-50 candidate_ids
     you've personally read and tagged with a relevance tier 0-3 (0 = honeypot/
     no-fit, 3 = excellent). Put them in data/my_labels.json as {id: tier}.
  2. Run your ranker, then call this to compute NDCG@10, NDCG@50, MAP, P@10
     against YOUR labels. This is how you tune WEIGHTS with evidence instead of
     vibes — and it's exactly the kind of rigor the JD (and Stage 5) rewards.

This file implements the same metrics the organizers use so your local numbers
are directionally comparable (same definitions, different — smaller — label set).

Metrics implemented:
  - DCG / NDCG@k   (graded relevance, standard log2 discount)
  - MAP            (treats tier >= relevance_threshold as relevant)
  - P@k            (precision at k, tier >= 3 counts as relevant per the spec)
"""

import json
import math
from typing import Dict, List

RELEVANT_TIER = 3   # spec: P@10 counts tier 3+ as "relevant"


def dcg_at_k(gains: List[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked_tiers: List[int], k: int) -> float:
    actual = dcg_at_k(ranked_tiers, k)
    ideal_order = sorted(ranked_tiers, reverse=True)
    ideal = dcg_at_k(ideal_order, k)
    return actual / ideal if ideal > 0 else 0.0


def average_precision(ranked_tiers: List[int], threshold: int = RELEVANT_TIER) -> float:
    hits = 0
    precisions = []
    for i, t in enumerate(ranked_tiers, start=1):
        if t >= threshold:
            hits += 1
            precisions.append(hits / i)
    return sum(precisions) / len(precisions) if precisions else 0.0


def precision_at_k(ranked_tiers: List[int], k: int, threshold: int = RELEVANT_TIER) -> float:
    top = ranked_tiers[:k]
    return sum(1 for t in top if t >= threshold) / k if k else 0.0


def evaluate(submission_csv: str, labels_path: str) -> Dict[str, float]:
    """Score a submission CSV against hand labels. Only ranked candidates that
    appear in your label set contribute; the rest are skipped (you can't grade
    what you haven't labeled)."""
    with open(labels_path) as f:
        labels: Dict[str, int] = json.load(f)

    import csv
    ranked_tiers = []
    with open(submission_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["candidate_id"]
            if cid in labels:
                ranked_tiers.append(labels[cid])

    if not ranked_tiers:
        raise SystemExit(
            "None of your ranked candidates are in data/my_labels.json — "
            "label some of your top picks first."
        )

    composite = (
        0.50 * ndcg_at_k(ranked_tiers, 10) +
        0.30 * ndcg_at_k(ranked_tiers, 50) +
        0.15 * average_precision(ranked_tiers) +
        0.05 * precision_at_k(ranked_tiers, 10)
    )
    return {
        "n_labeled_in_ranking": len(ranked_tiers),
        "NDCG@10": round(ndcg_at_k(ranked_tiers, 10), 4),
        "NDCG@50": round(ndcg_at_k(ranked_tiers, 50), 4),
        "MAP": round(average_precision(ranked_tiers), 4),
        "P@10": round(precision_at_k(ranked_tiers, 10), 4),
        "composite_estimate": round(composite, 4),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="output/submission.csv")
    ap.add_argument("--labels", default="data/my_labels.json")
    args = ap.parse_args()
    for k, v in evaluate(args.submission, args.labels).items():
        print(f"{k}: {v}")
