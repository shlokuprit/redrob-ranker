"""
rank.py
=======
THE RANKING STEP. This is the single command the organizers reproduce at Stage 3,
inside a sandboxed container (5 min, 16 GB, CPU, no network). It must:

  - load precomputed embeddings + meta (built earlier by embed.py)
  - embed ONLY the JD (one short string — cheap, allowed, no candidate API calls)
  - compute semantic similarity (one matmul)
  - combine with structured features (scorer.py)
  - write the top-100 CSV with grounded reasoning

Usage (matches submission_metadata.yaml reproduce_command):
    python src/rank.py --candidates data/candidates.jsonl --out output/submission.csv

If artifacts are missing, it tells you to run embed.py first. We embed the JD at
rank time (not precompute) so JD edits take effect without re-encoding 100k
candidates — and one tiny local encode call is well within budget.

NOTE: embedding the single JD string uses the local model. That's CPU-only and
needs no network once the model is cached locally. If you want a truly
network-free, model-free rank step, you can also precompute and store the JD
vector in artifacts/ — see the README "fully offline" note.
"""

import argparse
import csv
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
    JD_TEXT_FOR_EMBEDDING,
    OUTPUT_DIR,
    TOP_N,
)
from reasoning import build_reasoning
from scorer import compute_scores, rank_top_n


def load_artifacts():
    if not EMBEDDINGS_PATH.exists():
        raise SystemExit(
            f"Missing {EMBEDDINGS_PATH}. Run:  python src/embed.py "
            f"--candidates data/candidates.jsonl   (the precompute step)"
        )
    emb = np.load(EMBEDDINGS_PATH)
    with open(CAND_IDS_PATH) as f:
        ids = json.load(f)
    with open(CAND_META_PATH, "rb") as f:
        meta = pickle.load(f)
    assert len(ids) == len(meta) == emb.shape[0], "artifact length mismatch"
    return emb, ids, meta


def embed_jd():
    """Encode the single JD string with the same local model + normalization."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL_NAME)
    v = model.encode(
        [JD_TEXT_FOR_EMBEDDING], convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)
    return v[0]   # (EMBED_DIM,)


def main(candidates_path, out_path):
    t0 = time.time()
    print("Loading precomputed artifacts...")
    emb, ids, meta = load_artifacts()
    print(f"  {emb.shape[0]} candidates, dim {emb.shape[1]}")

    print("Embedding the job description...")
    jd_vec = embed_jd()

    print("Computing semantic similarity (matmul)...")
    # both sides are unit-normalized, so dot product == cosine similarity
    semantic_sim = emb @ jd_vec          # (N,)

    print("Combining with structured features...")
    final = compute_scores(semantic_sim, meta)

    print(f"Ranking top {TOP_N}...")
    top = rank_top_n(final, ids, TOP_N)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (row_idx, cid, score) in enumerate(top, start=1):
            reasoning = build_reasoning(meta[row_idx], score)
            w.writerow([cid, rank, f"{score:.4f}", reasoning])

    print(f"Wrote {out_path}")
    print(f"Ranking step completed in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(CANDIDATES_PATH))
    ap.add_argument("--out", default=str(OUTPUT_DIR / "submission.csv"))
    args = ap.parse_args()
    main(args.candidates, args.out)
