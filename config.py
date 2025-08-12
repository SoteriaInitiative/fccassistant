# ---- USER / PROJECT ----
PROJECT_ID = "fcc-assistant-mvp"
LOCATION = "us-central1"
GCS_CORPUS_BUCKET = "fcc-assistant-84"  # gs://fcc-assistant-mvp-42
WORKDIR = ".cache_corpus"  # local scratch download
LOCAL_DATA = "data"

# ---- TUNING ----
# Put the training JSONL under your existing bucket in a /tuning/ prefix.
# No hardcoding in the script — we derive this from the bucket.
TUNING_PREFIX = "tuning"
TUNING_DISPLAY_NAME = "fccassistant-ofac-fatf-gemini-sft"
OUT_NAME = "tuning_dataset_contents.jsonl"
MAX_PAIRS = 1500            # cap to keep runs cheap
MIN_CHARS = 200            # skip tiny chunks
MAX_MODEL_CHARS = 800      # cap the "ideal answer" text
INSTRUCTION = "Summarize the following accurately and concisely."
TUNING_BASE_MODEL = "gemini-2.5-flash"  # small/cheap to tune
EPOCHS = 3

# ---- EMBEDDING ----
# Embeddings / index
EMBEDDING_MODEL = "text-embedding-004"  # or your chosen Vertex embedding model
EMBEDDING_OUT_VECS = "embeddings.npy"
EMBEDDING_OUT_META = "embeddings_meta.jsonl"
FAISS_INDEX_NAME   = "faiss.index"
EMBED_BATCH_SIZE   = 32
EMBED_REQ_MAX_TOKENS = 20000
EMBED_REQ_SAFETY = 500


# ---- CHUNKING ----
KNOWLEDGE_SOURCE="knowledge"
CHUNK_SIZE = 1200
CHUNK_MAX_SIZE = 15000
CHUNK_OVERLAP = 150

# ---- VALIDATION SETTINGS ----
VALIDATION_IN = "validation_contents.jsonl"
VALIDATION_PREFIX = "validation"
VALIDATION_PROMPTS = 100
VALIDATION_INPUT_TOKENS_PER_PROMPT = 1000
VALIDATION_OUTPUT_TOKENS_PER_PROMPT = 500
EPOCHS = 3

# ---- COST CONSTANTS (paste from current Vertex pricing page) ----
# Units are USD.
# Embeddings: price PER 1,000 TOKENS (input).
COST_EMBEDDING_PER_1K_TOKENS = None  # e.g., 0.02
# Gemini tuning: price PER 1,000,000 TRAINING TOKENS.
COST_TUNING_PER_1M_TOKENS = None     # e.g., 5.00
# Validation inference (Gemini Flash/Flash-Lite): input/output per 1,000 TOKENS.
COST_VALIDATION_INPUT_PER_1K = None  # e.g., 0.10
COST_VALIDATION_OUTPUT_PER_1K = None # e.g., 0.40

# ---- SAFETY GUARDS ----
MAX_ALLOWED_ESTIMATED_COST_USD = 5.00  # refuse to run if estimate exceeds this
