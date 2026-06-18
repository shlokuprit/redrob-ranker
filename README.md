# Redrob Intelligent Candidate Ranking — India.Runs 2026 (Data & AI Challenge)

Semantic + hybrid candidate ranker for the Redrob "Senior AI Engineer" JD. Ranks
100,000 candidates and outputs the top 100 a recruiter can trust — by reasoning
about **what the JD means**, not by counting AI keywords.

## TL;DR architecture

```
candidates.jsonl ──> embed.py (PRECOMPUTE, slow, allowed)
                       │  - text per candidate (data_io.candidate_to_text)
                       │  - MiniLM sentence embeddings (normalized)
                       │  - structured features (features.py)
                       └─> artifacts/  {embeddings.npy, ids.json, meta.pkl}

artifacts/ + JD ──> rank.py (RANKING STEP, <5 min, CPU, no network)
                       │  - embed the single JD string
                       │  - semantic_sim = embeddings @ jd_vec   (one matmul)
                       │  - final = fit(weighted sum) × behavior × consulting × plausibility
                       │  - top-100, grounded reasoning
                       └─> output/submission.csv
```

The scoring formula:

```
fit   = 0.40·semantic + 0.22·skills + 0.18·title + 0.12·exp + 0.05·location + 0.03·education
final = fit × behavior_multiplier × consulting_multiplier × plausibility_multiplier
```

- **semantic** — cosine similarity of JD vs candidate meaning-vectors (the "reads
  the profile like a recruiter" part).
- **title** — the decisive anti-keyword-stuffing signal. A Marketing Manager with
  every AI buzzword scores low here on purpose.
- **plausibility** — quietly sinks the ~80 honeypots (internally-impossible
  profiles) without hand-labeling them. Keeps honeypot rate in the top-100 low,
  which is a hard Stage-3 disqualifier if it exceeds 10%.
- **behavior** — collapses the 23 Redrob signals into one bounded multiplier:
  great-on-paper but unreachable ⇒ down-weighted, not removed.

Every weight and rule lives in `src/config.py` with the reasoning in comments —
that's the file to point at when defending design choices at Stage 5.

## Repo layout

```
src/
  config.py     all tunable params + the structured JD (read the comments)
  data_io.py    streaming loader + candidate->text serialization
  features.py   structured signals, behavior multiplier, honeypot detection
  embed.py      PRECOMPUTE: build candidate embeddings + meta -> artifacts/
  scorer.py     vectorized hybrid scoring + ranking/tie-break
  reasoning.py  grounded, non-templated reasoning strings (no LLM, no hallucination)
  rank.py       THE reproduce command -> output/submission.csv
  evaluate.py   local NDCG@10/@50/MAP/P@10 against your own hand labels
tests/
  test_smoke.py tiny end-to-end check (good > stuffer > honeypot)
data/           put candidates.jsonl here (gitignored; it's huge)
artifacts/      precomputed embeddings/meta (gitignored or via Git LFS)
output/         submission.csv lands here
docs/           deck source / notes
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# CPU torch wheel (smaller, matches the no-GPU constraint):
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
```

Put the candidate pool at `data/candidates.jsonl` (gunzip the provided `.gz`).

## Run

```bash
# 1) PRECOMPUTE (one-time, may exceed 5 min — that's allowed for precompute)
python src/embed.py --candidates data/candidates.jsonl

# 2) RANK (this is the step reproduced at Stage 3; must finish < 5 min on CPU)
python src/rank.py --candidates data/candidates.jsonl --out output/submission.csv

# 3) VALIDATE format before submitting
python validate_submission.py output/submission.csv

# 4) (optional) score against your own labels to tune weights
python src/evaluate.py --submission output/submission.csv --labels data/my_labels.json
```

**Reproduce command** (also in `submission_metadata.yaml`):

```
python src/rank.py --candidates ./data/candidates.jsonl --out ./output/submission.csv
```

## Compute constraints — how we satisfy them

| Constraint            | How we meet it                                                        |
|-----------------------|----------------------------------------------------------------------|
| ≤ 5 min ranking       | embeddings precomputed; rank = 1 matmul + vectorized numpy           |
| ≤ 16 GB RAM           | 100k × 384 float32 ≈ 150 MB matrix; meta is small                    |
| CPU only              | no GPU code anywhere; MiniLM runs on CPU                             |
| network off at rank   | no hosted-LLM calls; only a local model encodes the one JD string    |
| ≤ 5 GB disk           | artifacts are a few hundred MB                                       |

> **Fully-offline option:** to make `rank.py` need *no* model at all, precompute
> the JD vector during `embed.py` and store it in `artifacts/jd_vec.npy`, then
> load it in `rank.py`. See the note at the top of `rank.py`.

## Honeypot handling

We never hard-code candidate IDs. `features.implausibility_penalty` flags
internal contradictions (expert skill with 0 months used; a skill claiming more
months than the whole career; role longer than the entire stated career;
career-duration vs stated-experience mismatch). These mirror the examples in
`submission_spec.md §7` and push honeypots down naturally.

## Building it across sessions

This scaffold is intentionally modular so each work session has a clear target:

1. ✅ Scaffold + smoke test (done)
2. Run `embed.py` on the real data, sanity-check the top 100
3. Read ~40 candidates by hand, write `data/my_labels.json`, tune `WEIGHTS`
   against `evaluate.py`
4. Inspect honeypots/stuffers in your top 100; refine `features.py`
5. Build the deck (`docs/`), the HuggingFace/Streamlit sandbox, fill metadata
6. Final validate + submit

## AI tools

Declared in `submission_metadata.yaml`. Claude was used for architecture
discussion and code review; the engineering, tuning, and design decisions are the
team's own. No candidate data was sent to any hosted LLM, and the ranker makes no
LLM calls.
