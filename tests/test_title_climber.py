"""
test_title_climber.py
=====================
Proves the title-chaser detector fires ONLY on the genuine JD pattern
(short tenure + ascending seniority), not on lateral movement or healthy growth.

These four cases are the REAL candidates from the top-100 we inspected, plus one
synthetic true-climber to prove the detector can fire when the pattern is real.
Run: python tests/test_title_climber.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import features as F


def mk(history):
    return {"career_history": [{"title": t, "duration_months": m} for t, m in history]}


def run():
    # Real lateral mover (rank-5): all base-level, different domains -> NOT a climber
    lateral = mk([("Recommendation Systems Engineer",14),("Search Engineer",16),
                  ("NLP Engineer",27),("Applied ML Engineer",13)])
    # Real erratic bouncer (CAND_0005538): Lead<->Senior, ends lower -> NOT a climber
    erratic = mk([("Senior AI Engineer",15),("Lead AI Engineer",30),
                  ("Senior Machine Learning Engineer",14),("Lead AI Engineer",10)])
    # Real healthy growth (CAND_0047521): Junior->Senior once -> NOT a chaser
    healthy = mk([("Senior Software Engineer (ML)",15),
                  ("Senior Software Engineer (ML)",20),("Junior ML Engineer",16)])
    # SYNTHETIC true climber: Junior->Senior->Lead->Principal, all short -> IS a climber
    climber = mk([("Principal ML Engineer",12),("Lead ML Engineer",14),
                  ("Senior ML Engineer",13),("Junior ML Engineer",15)])

    assert F.title_climber_penalty(lateral) == 1.0, "lateral mover wrongly penalized"
    assert F.title_climber_penalty(erratic) == 1.0, "erratic bouncer wrongly penalized"
    assert F.title_climber_penalty(healthy) == 1.0, "healthy growth wrongly penalized"
    assert F.title_climber_penalty(climber) == 0.93, "true climber NOT caught"

    print("lateral mover   -> ", F.title_climber_penalty(lateral), "(expect 1.0)")
    print("erratic bouncer -> ", F.title_climber_penalty(erratic), "(expect 1.0)")
    print("healthy growth  -> ", F.title_climber_penalty(healthy), "(expect 1.0)")
    print("true climber    -> ", F.title_climber_penalty(climber), "(expect 0.93)")
    print("\nAll title-climber assertions passed.")


if __name__ == "__main__":
    run()
