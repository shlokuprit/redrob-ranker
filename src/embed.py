"""
embed.py
========
PRECOMPUTE STEP (allowed to exceed the 5-min budget).

Builds a (N x EMBED_DIM) matrix of candidate embeddings and saves it to
artifacts/. The ranking step (rank.py) then just loads this matrix — no model
inference per candidate at rank time, which is how we stay inside the 5-minute /
CPU / no-network constraint for the part that actually gets reproduced.

Run once:
    python src/embed.py --candidates data/candidates.jsonl

Outputs:
    artifacts/cand_embeddings.npy   float32 (N, 384)
    artifacts/cand_ids.json         list[str], row order matches the matrix
    artifacts/cand_meta.pkl         per-candidate precomputed structured features

Design notes for the interview:
  - We normalize embeddings to unit length so cosine similarity is just a dot
    product (fast, and lets us batch the JD-vs-all comparison as one matmul).
  - We also precompute the *structured* features here and pickle them, so rank.py
    does almost no per-candidate Python work — it's mostly vectorized numpy.
"""

import argparse
import json
import pickle
import time

import numpy as np

from config import (
    CAND_IDS_PATH,
    CAND_META_PATH,
    CANDIDATES_PATH,
    EMBED_MODEL_NAME,
    EMBEDDINGS_PATH,
)
from data_io import candidate_to_text, stream_candidates
import features as F


def build(candidates_path, batch_size=512):
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    ids, texts, meta = [], [], []
    t0 = time.time()

    for i, c in enumerate(stream_candidates(candidates_path)):
        ids.append(c["candidate_id"])
        texts.append(candidate_to_text(c))
        # precompute every structured feature ONCE here
        meta.append({
            "skills": F.skills_score(c),
            "title": F.title_score(c),
            "exp": F.experience_score(c),
            "location": F.location_score(c),
            "education": F.education_score(c),
            "consulting": F.consulting_only_penalty(c),
            "behavior": F.behavior_multiplier(c),
            "plausibility": F.implausibility_penalty(c),
            # keep a few raw fields for building the reasoning string later
            "current_title": c.get("profile", {}).get("current_title", ""),
            "yoe": c.get("profile", {}).get("years_of_experience"),
            "resp_rate": c.get("redrob_signals", {}).get("recruiter_response_rate"),
            "last_active": c.get("redrob_signals", {}).get("last_active_date"),
        })
        if (i + 1) % 10000 == 0:
            print(f"  prepared {i+1} candidates ({time.time()-t0:.1f}s)")

    print(f"Encoding {len(texts)} candidates (this is the slow part)...")
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit vectors => cosine == dot product
    ).astype(np.float32)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, emb)
    with open(CAND_IDS_PATH, "w") as f:
        json.dump(ids, f)
    with open(CAND_META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"Saved embeddings {emb.shape} -> {EMBEDDINGS_PATH}")
    print(f"Saved ids -> {CAND_IDS_PATH}")
    print(f"Saved meta -> {CAND_META_PATH}")
    print(f"Total precompute time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(CANDIDATES_PATH))
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()
    build(args.candidates, args.batch_size)
