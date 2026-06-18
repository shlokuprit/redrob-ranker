"""
label_helper.py
===============
Fast hand-labeling helper. Shows ~12 candidates pulled from across your ranking
(top, middle, low, plus flagged) in compact form, and collects a relevance tier
(0-3) for each. Writes data/my_labels.json for use by src/evaluate.py.

Spread matters: labeling only top candidates can't measure whether your ranking
ORDERS candidates correctly. Including mid/low candidates lets NDCG see separation.

Tiers:
  3 = excellent fit (deep relevant experience, on-target, plausible)
  2 = good fit (solid, some gaps)
  1 = marginal (limited relevance / notable gaps)
  0 = no fit / wrong role / honeypot-like

Run from project root (venv active):
    python label_helper.py
"""
import json, csv, sys

CANDIDATES = "data/candidates.jsonl"
SUBMISSION = "output/submission.csv"
OUT = "data/my_labels.json"


def load_candidates():
    return {json.loads(l)["candidate_id"]: json.loads(l)
            for l in open(CANDIDATES, encoding="utf-8")}


def compact(c):
    p = c.get("profile", {})
    lines = []
    lines.append(f"  Title: {p.get('current_title','?')}  |  YOE: {p.get('years_of_experience','?')}  |  Loc: {p.get('location','?')}")
    summ = (p.get("summary") or "").strip().replace("\n", " ")
    if summ:
        lines.append(f"  Summary: {summ[:240]}")
    # recent roles
    hist = c.get("career_history", [])[:3]
    if hist:
        roles = "; ".join(f"{r.get('title','?')} ({r.get('duration_months','?')}mo)" for r in hist)
        lines.append(f"  Recent roles: {roles}")
    # top skills by duration
    sk = sorted(c.get("skills", []), key=lambda s: -(s.get("duration_months", 0) or 0))[:8]
    if sk:
        skl = ", ".join(f"{s.get('name','?')}({s.get('duration_months','?')}mo)" for s in sk)
        lines.append(f"  Top skills: {skl}")
    return "\n".join(lines)


def pick_spread(ranked_ids, n_total=12):
    """Pick a spread across the ranking: top, upper-mid, mid, low."""
    N = len(ranked_ids)
    idxs = sorted(set([
        0, 1, 2,                       # top 3
        N//8, N//4,                    # upper area
        N//2, N//2 + 1,                # middle
        3*N//4,                        # lower-mid
        N-3, N-2, N-1,                 # bottom of top-100
    ]))
    picks = [ranked_ids[i] for i in idxs if i < N]
    return picks[:n_total]


def main():
    cand = load_candidates()
    ranked = [r["candidate_id"] for r in csv.DictReader(open(SUBMISSION, encoding="utf-8"))]
    picks = pick_spread(ranked)

    print("="*70)
    print("HAND-LABELING — assign each candidate a tier:")
    print("  3=excellent  2=good  1=marginal  0=no-fit/wrong-role")
    print("(These are shown WITHOUT their rank so your judgment stays independent.)")
    print("="*70)

    labels = {}
    for n, cid in enumerate(picks, 1):
        c = cand.get(cid)
        if not c:
            continue
        print(f"\n--- Candidate {n} of {len(picks)} ---")
        print(compact(c))
        while True:
            ans = input("  Your tier [0/1/2/3] (or q to quit): ").strip().lower()
            if ans == "q":
                print("Aborted; nothing saved.")
                sys.exit(0)
            if ans in {"0", "1", "2", "3"}:
                labels[cid] = int(ans)
                break
            print("  Please enter 0, 1, 2, 3, or q.")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    print(f"\nSaved {len(labels)} labels -> {OUT}")
    print("Now run:  python src/evaluate.py --submission output/submission.csv --labels data/my_labels.json")


if __name__ == "__main__":
    main()
