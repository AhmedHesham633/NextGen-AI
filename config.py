"""
Central configuration for the Clinical RAG project.
Edit these values to match your team's setup — everything else
in this repo reads from here, so you only need to change it in one place.
"""
import os
from pathlib import Path
from dotenv import load_dotenv # type: ignore

load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "clinical_guidelines"

# --- Chunking ---
# Values are in approximate tokens. The splitter uses a rough
# 4-characters-per-token estimate to convert these to character counts.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 180

# --- Embeddings ---
# "local"  -> free, runs on your machine, lightweight, no API key needed (default)
# "openai" -> optional, requires OPENAI_API_KEY in .env
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# --- Retrieval ---
TOP_K = 10

# --- Large File Limits ---
# Set to an integer to sample large CSV datasets, or None to load all rows.
MAX_CSV_ROWS = 3000

# --- Confidence Score Thresholds (tuned for BAAI/bge-small-en-v1.5 cosine similarity) ---
# Scores typically range 0.40 – 0.80 with local embeddings
CONFIDENCE_HIGH_THRESHOLD   = 0.65   # score >= 0.65 → HIGH
CONFIDENCE_MEDIUM_THRESHOLD = 0.50   # score >= 0.50 → MEDIUM
CONFIDENCE_LOW_THRESHOLD    = 0.38   # score >= 0.38 → LOW
                                      # score <  0.38 → INSUFFICIENT

# --- Generation (Gemini) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


