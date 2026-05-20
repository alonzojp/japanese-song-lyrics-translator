import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CACHE_DIR = Path(os.getenv("CACHE_DIR", BASE_DIR / "cache"))
DB_PATH = Path(os.getenv("DB_PATH", CACHE_DIR / "jobs.db"))

# ── Worker ─────────────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "1"))

# ── Whisper ────────────────────────────────────────────────────────────────────
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # auto | cpu | cuda
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")  # auto | int8 | float16 | float32
WHISPER_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "16"))
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ja")

# ── Demucs ─────────────────────────────────────────────────────────────────────
DEMUCS_MODEL = os.getenv("DEMUCS_MODEL", "htdemucs")
SKIP_VOCAL_SEPARATION = os.getenv("SKIP_VOCAL_SEPARATION", "false").lower() == "true"

# ── Cache ──────────────────────────────────────────────────────────────────────
# Keep audio files after processing (saves re-downloading)
KEEP_AUDIO = os.getenv("KEEP_AUDIO", "true").lower() == "true"

# ── Ensure dirs exist ──────────────────────────────────────────────────────────
CACHE_DIR.mkdir(parents=True, exist_ok=True)
