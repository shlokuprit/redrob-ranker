# Deck outline + build plan

## Submission deck (export to PDF). Suggested 10–12 slides:

1. **Title** — team, track, one-line: "Semantic + hybrid ranking that reasons
   about JD intent, not keywords."
2. **The problem** — rank 100k candidates for one nuanced JD; keyword matching
   fails; honeypots and keyword-stuffers are traps.
3. **Key insight** — the gap between what the JD *says* and what it *means*.
   Title beats keyword count; availability matters; impossible profiles exist.
4. **Architecture diagram** — precompute (embeddings + features) → rank step
   (matmul + hybrid scoring). Show the 5-min CPU constraint is met by precompute.
5. **Semantic layer** — what text we embed per candidate and why order matters
   (summary + recent role descriptions surface "Tier-5" hidden fits).
6. **Hybrid scoring** — the formula; what each component catches; the weights.
7. **Anti-trap design** — title signal vs keyword stuffers; plausibility penalty
   vs honeypots (show the smoke-test example: good > stuffer > honeypot).
8. **Behavioral signals** — 23 → 1 bounded multiplier; availability philosophy.
9. **Reasoning generation** — grounded, non-templated, no LLM ⇒ no hallucination,
   satisfies Stage-4 checks.
10. **Evaluation** — how you tuned weights with hand labels + NDCG/MAP locally
    (no live leaderboard, so methodology rigor is the story).
11. **Reproducibility & constraints** — the table; single reproduce command;
    sandbox link.
12. **What we'd do next** — learning-to-rank on labeled data, cross-encoder
    re-rank of top-K, calibration.

## Multi-session build plan

- [x] Session 1: scaffold, config, features, scorer, reasoning, rank, smoke test
- [ ] Session 2: run embed.py on real data; eyeball top 100; spot-check honeypots
- [ ] Session 3: hand-label ~40 candidates; tune WEIGHTS against evaluate.py
- [ ] Session 4: refine features (title lists, honeypot rules) from observed misses
- [ ] Session 5: build deck (this outline) + sandbox (Streamlit/HF Spaces)
- [ ] Session 6: final validate, fill metadata, submit (3 submissions max!)

## Things not to forget (from the spec)

- Filename must be `<participant_id>.csv`.
- Exactly 100 rows, ranks 1–100 unique, score non-increasing, IDs exist & unique.
- Honeypot rate in top 100 must stay ≤ 10% (Stage-3 disqualifier).
- Reasoning: specific, honest about concerns, varied, no hallucinated skills.
- Keep real git history (commit per session) — flat history is a Stage-4 flag.
- Sandbox link is mandatory; runs on ≤100-candidate sample in ≤5 min.
