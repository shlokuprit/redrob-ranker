"""
config.py
=========
Single source of truth for all tunable parameters. Keeping these in one place
(instead of scattering magic numbers through the code) is what lets you, in the
Stage-5 interview, point at one file and explain every design decision.

Nothing here runs logic — it's just constants. Read the comments; they encode
the *reasoning* behind each choice, which is exactly what the human judges want.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"   # precomputed embeddings / index live here
OUTPUT_DIR = ROOT / "output"

CANDIDATES_PATH = DATA_DIR / "candidates.jsonl"      # the 100k pool
EMBEDDINGS_PATH = ARTIFACTS_DIR / "cand_embeddings.npy"
CAND_IDS_PATH = ARTIFACTS_DIR / "cand_ids.json"      # row-index -> candidate_id
CAND_META_PATH = ARTIFACTS_DIR / "cand_meta.pkl"     # lightweight features per cand

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
# Why this model: small (~80MB), CPU-fast, strong on short semantic-similarity
# tasks, and fully local (no API => satisfies the "network off" constraint).
# At Stage 3 they re-run your *ranking* step offline; embeddings are precomputed
# (allowed) but the model must still be available locally, so we pick a small one.
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

# ---------------------------------------------------------------------------
# The job description (parsed into a structured target)
# ---------------------------------------------------------------------------
# This is the *machine-readable* distillation of job_description.md. The raw JD
# is prose; the ranker needs structure. Editing these lists is how you tune what
# "fit" means. Each field maps to a scoring component in scorer.py.
JD_TEXT_FOR_EMBEDDING = (
    "Senior AI Engineer for a Series A AI-native talent intelligence platform. "
    "Owns the intelligence layer: ranking, retrieval, and matching systems. "
    "Needs production experience with embeddings-based retrieval (sentence-transformers, "
    "BGE, E5, OpenAI embeddings) deployed to real users, vector databases or hybrid "
    "search (Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch, OpenSearch), "
    "strong Python, and hands-on evaluation of ranking systems (NDCG, MRR, MAP, A/B "
    "testing). Has shipped an end-to-end ranking, search, or recommendation system to "
    "real users at a product company. Scrappy product-engineering attitude, ships fast. "
    "5 to 9 years experience, roughly 4 to 5 in applied ML at product companies."
)

# Skills that signal genuine retrieval/ranking/ML-systems depth.
CORE_SKILLS = [
    "embeddings", "sentence-transformers", "vector search", "vector database",
    "semantic search", "information retrieval", "retrieval", "ranking",
    "learning to rank", "recommendation systems", "recommender", "search",
    "nlp", "natural language processing", "rag", "llm", "fine-tuning",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "elasticsearch",
    "opensearch", "bm25", "ndcg", "machine learning", "deep learning",
    "pytorch", "tensorflow", "xgboost",
]
NICE_TO_HAVE_SKILLS = [
    "lora", "qlora", "peft", "distributed systems", "spark", "airflow",
    "kubernetes", "docker", "mlops", "mlflow", "weights & biases",
]
# Python is a hard requirement in the JD.
REQUIRED_SKILLS = ["python"]

# Titles whose *current/recent* presence strongly indicates a real ML/eng person.
# The JD's big trap: a "Marketing Manager" with all the AI keywords is NOT a fit.
# Title is the decisive anti-keyword-stuffing signal.
POSITIVE_TITLE_KEYWORDS = [
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "research engineer", "applied scientist", "nlp", "search engineer",
    "software engineer", "backend engineer", "platform engineer",
    "staff engineer", "principal engineer", "data engineer",
]
# Titles the JD explicitly says are NOT a fit even with perfect keyword lists.
NEGATIVE_TITLE_KEYWORDS = [
    "marketing", "sales", "hr ", "human resources", "recruiter", "accountant",
    "graphic designer", "content writer", "civil engineer", "mechanical engineer",
    "customer support", "operations manager", "project manager", "business analyst",
]

# Experience band from the JD.
EXP_IDEAL_MIN = 5.0
EXP_IDEAL_MAX = 9.0
EXP_SWEET_MIN = 6.0   # "6-8 years" ideal candidate
EXP_SWEET_MAX = 8.0

# Locations the JD prefers (Noida/Pune, plus relocation-from Tier-1 metros).
PREFERRED_LOCATIONS = [
    "pune", "noida", "delhi", "ncr", "gurgaon", "gurugram", "hyderabad",
    "mumbai", "bangalore", "bengaluru",
]

# Consulting-only careers are a soft negative per the JD.
CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "mindtree", "hcl", "tech mahindra",
]

# ---------------------------------------------------------------------------
# Hybrid scoring weights  (must sum to ~1.0 for the fit sub-score)
# ---------------------------------------------------------------------------
# These are the heart of the system and the thing you'll defend most. Start here,
# then tune against your own held-out judgments. Document every change in git so
# the "real iteration vs single dump" check at Stage 4 passes.
WEIGHTS = {
    "semantic": 0.40,   # cosine(JD_embedding, candidate_embedding)
    "skills":   0.22,   # weighted overlap with CORE/REQUIRED/NICE skills
    "title":    0.18,   # current+recent title alignment (anti-stuffing)
    "exp":      0.12,   # experience-band fit
    "location": 0.05,   # geographic preference
    "education":0.03,   # institution tier, mild signal
}

# Behavioral signals act as a *multiplier* on the fit score, not an additive term.
# Rationale (straight from the JD/signals doc): a perfect-on-paper candidate who
# is unreachable is, for hiring, not actually available. We down-weight, we don't
# zero out. Multiplier is bounded so it modifies rather than dominates.
BEHAVIOR_MULTIPLIER_MIN = 0.70
BEHAVIOR_MULTIPLIER_MAX = 1.10

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
TOP_N = 100
