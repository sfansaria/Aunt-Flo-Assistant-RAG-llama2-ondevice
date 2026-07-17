import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM backend ---
# Point at local Ollama by default. Swap to a hosted provider by changing
# LLM_BASE_URL / LLM_API_KEY / LLM_MODEL in .env — no code changes needed.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")

# --- Embeddings ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# --- Reranker ---
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"

# --- Vector store ---
QDRANT_PATH = os.getenv("QDRANT_PATH", "./vectorstore/qdrant_db")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "auntflo")

# --- BM25 sparse index (pickled alongside the vector store) ---
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "./vectorstore/bm25.pkl")

# --- Retrieval tuning ---
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "8"))   # candidates before rerank
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "4"))           # chunks sent to the LLM
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# --- Session memory ---
SESSION_DB_PATH = os.getenv("SESSION_DB_PATH", "./sessions.db")
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))

# --- Data ---
PDF_DIR = os.getenv("PDF_DIR", "./data")

SYSTEM_PROMPT = """You are Aunt Flo Assistant, a friendly, knowledgeable guide on menstrual and \
reproductive health. Answer using ONLY the provided context. Cite sources inline like [1], [2] \
matching the numbered context passages. If the context doesn't contain the answer, say so plainly \
rather than guessing.

You are not a substitute for professional medical advice. For anything urgent, unusual, or \
personal to someone's specific health situation, encourage them to see a doctor or qualified \
provider."""

EMERGENCY_KEYWORDS = [
    "suicidal", "want to die", "kill myself", "self harm", "self-harm",
    "severe bleeding", "soaking a pad every hour", "chest pain", "can't breathe",
    "passed out", "fainted", "severe abdominal pain",
]

EMERGENCY_RESPONSE = (
    "This sounds like it could be urgent. Please contact a doctor, urgent care, or emergency "
    "services right away rather than relying on this assistant. If you're in the US, you can call "
    "911 for a medical emergency or 988 for a mental health crisis."
)
