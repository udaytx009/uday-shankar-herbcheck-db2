"""Settings, read from environment variables (or a local .env file)."""
import os

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

DB2 = {
    "host": os.getenv("DB2_HOST", "localhost"),
    "port": os.getenv("DB2_PORT", "50000"),
    "database": os.getenv("DB2_DATABASE", "herbdb"),
    "user": os.getenv("DB2_USER", "db2inst1"),
    "password": os.getenv("DB2_PASSWORD", ""),
    "security": os.getenv("DB2_SECURITY", "false").lower() == "true",
}

EMBED_DIM = 384  # BAAI/bge-small-en-v1.5 and the built-in hashing embedder both produce 384 dims
EMBEDDINGS = os.getenv("EMBEDDINGS", "auto").lower()
_sd = os.getenv("SUGGEST_MAX_DIST", "").strip()
SUGGEST_MAX_DIST = float(_sd) if _sd else None  # None = use the embedder's own default
EVIDENCE_TABLE = "HERB_EVIDENCE_VS"  # table owned by langchain_db2.DB2VS

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()

MAX_MEDICATIONS = 6
MAX_HERBS = 5
MAX_NAME_LEN = 60
