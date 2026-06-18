"""
rebuild_meta.py
===============
Rebuilds ONLY artifacts/cand_meta.pkl from the candidate file, using the current
features.py logic. Use this after changing any feature function, so you don't have
to re-run the slow GPU embedding (embeddings are unaffected by feature changes).

Run from the project root:
    python rebuild_meta.py --candidates data/candidates.jsonl

Pure-Python feature computation over 100k candidates takes well under a minute on CPU.
It reads cand_ids.json to keep the row order IDENTICAL to the embeddings matrix.
"""
import argparse, json, pickle, time
import sys
sys.path.insert(0, "src")

from config import CANDIDATES_PATH, CAND_IDS_PATH, CAND_META_PATH
from data_io import stream_candidates
import features as F


def main(candidates_path):
    t0 = time.time()
    # load existing id order so meta rows line up with the embeddings matrix
    with open(CAND_IDS_PATH) as f:
        id_order = json.load(f)
    pos = {cid: i for i, cid in enumerate(id_order)}

    meta = [None] * len(id_order)
    seen = 0
    for c in stream_candidates(candidates_path):
        cid = c["candidate_id"]
        if cid not in pos:
            continue
        meta[pos[cid]] = {
            "skills": F.skills_score(c),
            "title": F.title_score(c),
            "exp": F.experience_score(c),
            "location": F.location_score(c),
            "education": F.education_score(c),
            "consulting": F.consulting_only_penalty(c),
            "title_climber": F.title_climber_penalty(c),
            "behavior": F.behavior_multiplier(c),
            "plausibility": F.implausibility_penalty(c),
            "current_title": c.get("profile", {}).get("current_title", ""),
            "yoe": c.get("profile", {}).get("years_of_experience"),
            "resp_rate": c.get("redrob_signals", {}).get("recruiter_response_rate"),
            "last_active": c.get("redrob_signals", {}).get("last_active_date"),
        }
        seen += 1
        if seen % 20000 == 0:
            print(f"  rebuilt {seen} ({time.time()-t0:.1f}s)")

    missing = [i for i, m in enumerate(meta) if m is None]
    if missing:
        raise SystemExit(f"ERROR: {len(missing)} ids in cand_ids.json not found in data")

    with open(CAND_META_PATH, "wb") as f:
        pickle.dump(meta, f)
    print(f"Rebuilt meta for {seen} candidates in {time.time()-t0:.1f}s -> {CAND_META_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(CANDIDATES_PATH))
    args = ap.parse_args()
    main(args.candidates)
