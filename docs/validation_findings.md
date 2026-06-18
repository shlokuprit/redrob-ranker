# Validation findings & design decisions

This document records analyses run against the real top-100 output, the issues
they surfaced, and the decisions made. It exists to demonstrate evidence-based
iteration (not guesswork) and to supply concrete talking points for the deck and
the live interview.

## 1. Honeypots are unlabeled — detection, not filtering

The candidate schema has no `is_honeypot` field; honeypots are hidden and must be
detected. Our `implausibility_penalty` flags internal contradictions. Verified by
inspecting the lowest-plausibility candidate (CAND_0019480, "NLP Engineer", 2.8 yrs
stated) which claimed 82 months of LLMs and 58 of Milvus — each longer than the
entire career, and roles summing to ~7.3 yrs against 2.8 stated. A pure
keyword/semantic matcher would rank this person highly; our plausibility multiplier
(0.55) kept them out of the top 100. **Honeypots in final top 100: 0.**

## 2. Title-chaser disqualifier — climbing-aware, soft penalty

JD rejects candidates "optimizing for Senior -> Staff -> Principal titles by
switching companies every 1.5 years." Naive tenure (<18mo) flagged 4 top-100
candidates, but reading them showed 3 were lateral movers (ML/Search/NLP/Rec at the
same level — actually a strength here) and 1 was an erratic Lead<->Senior bouncer,
not a true climber. Built `title_climber_penalty`: fires only on short tenure AND a
SUSTAINED upward ladder (>=2 upward seniority steps, net climb >=2, more ups than
downs). Tested against all 4 real cases + a synthetic true-climber. **True climbers
in top 100: 0.** Penalty kept soft (0.93x) because climbing is an inference, not a
certainty — penalty is calibrated to confidence.

## 3. Shallow-LLM-only disqualifier — verified clean, deliberately NOT penalized

JD rejects candidates whose AI experience is only recent LLM-API plumbing
(LangChain/OpenAI <12 months) with no real ML foundation. First detector flagged 10
top-100 candidates — but reading CAND_0077337 ("Staff ML Engineer", 7 yrs) revealed
the detector was broken: its "foundation" list omitted retrieval/ranking/rec-sys
skills, so a 7-year expert with 86mo of Recommendation Systems and 61mo of
Information Retrieval was falsely flagged as having "0 months foundation." Shipping
this would have demoted 10 of the strongest, most-relevant candidates.

Corrected the ML-depth definition to include retrieval, ranking, recommendation,
semantic search, NLP, classical ML, frameworks, and embedding skills. Re-ran:
**shallow-LLM-only in top 100 dropped from 10 to 0.** Conclusion: the existing
duration-weighted skills score already keeps shallow profiles out. Deliberately did
NOT add a redundant multiplier, to avoid the false-positive risk just demonstrated.

### Key engineering principle (interview talking point)
A detector is only as good as its definition. Every flag was validated against real
profiles BEFORE being allowed to affect output. This prevented a definition error
from silently demoting strong candidates — restraint and validation over piling on
penalties.

## 4. Weight validation via hand-labeled spot-check

Built `label_helper.py` to hand-label a SPREAD of candidates (top/mid/low of the
ranking, shown without their rank to keep judgment independent). Labeled 11
candidates on a 0-3 relevance scale, then scored the ranking against these labels
with `src/evaluate.py`:

  NDCG@10 = 0.95, NDCG@50 = 0.97, MAP = 0.78, P@10 = 0.60

Interpretation: the system's ordering agrees strongly with independent human
judgment (NDCG@10 ~0.95). P@10 is noisy at this label count and counts only tier-3
as relevant. These are DIRECTIONAL evidence (11 labels), not a competition score.

Decision: keep the principled weights as-is. With NDCG@10 ~0.95 against the labeled
sample, re-weighting would risk OVERFITTING to 11 noisy labels and could worsen
hidden performance. Knowing when not to tune is a deliberate choice here, not an
omission. Weights are set from the JD's stated priorities (semantic-dominant, with
title/skills/experience as the recruiter checks) and validated, not guessed.
