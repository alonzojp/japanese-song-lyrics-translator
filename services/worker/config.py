import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
CACHE_DIR = Path(os.getenv("CACHE_DIR", str(BASE_DIR / "cache")))
DB_PATH   = Path(os.getenv("DB_PATH",   str(CACHE_DIR / "jobs.db")))

# ── Worker ─────────────────────────────────────────────────────────────────────
HOST               = os.getenv("HOST", "0.0.0.0")
PORT               = int(os.getenv("PORT", "8000"))
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "1"))

# ── Whisper ────────────────────────────────────────────────────────────────────
WHISPER_MODEL        = os.getenv("WHISPER_MODEL",        "large-v3")
WHISPER_DEVICE       = os.getenv("WHISPER_DEVICE",       "auto")   # auto|cpu|cuda
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")   # auto|int8|float16|float32
WHISPER_BATCH_SIZE   = int(os.getenv("WHISPER_BATCH_SIZE", "16"))
WHISPER_LANGUAGE     = os.getenv("WHISPER_LANGUAGE", "ja")

# ── Demucs ─────────────────────────────────────────────────────────────────────
DEMUCS_MODEL           = os.getenv("DEMUCS_MODEL", "htdemucs")
SKIP_VOCAL_SEPARATION  = os.getenv("SKIP_VOCAL_SEPARATION", "false").lower() == "true"

# ── Audio processing ───────────────────────────────────────────────────────────
# Target loudness for EBU R128 normalisation (LUFS). -23 is broadcast standard.
AUDIO_LUFS_TARGET = float(os.getenv("AUDIO_LUFS_TARGET", "-23"))
# True peak ceiling (dBTP).
AUDIO_TRUE_PEAK   = float(os.getenv("AUDIO_TRUE_PEAK", "-1"))
# Sample rate for the normalised Demucs-input WAV (Hz). 44100 = Demucs default.
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "44100"))
# Trim leading/trailing silence from audio before processing.
TRIM_SILENCE      = os.getenv("TRIM_SILENCE", "true").lower() == "true"
# Silence threshold used by silenceremove (dB below which is "silence").
SILENCE_THRESHOLD = os.getenv("SILENCE_THRESHOLD", "-60dB")
# Minimum silence duration to trim (seconds).
SILENCE_DURATION  = float(os.getenv("SILENCE_DURATION", "0.5"))

# ── Cache ──────────────────────────────────────────────────────────────────────
# Delete cache dirs not accessed for longer than this many days.
CACHE_MAX_AGE_DAYS = int(os.getenv("CACHE_MAX_AGE_DAYS", "30"))
# Keep raw audio file after processing (saves re-downloading if stages are re-run).
KEEP_AUDIO = os.getenv("KEEP_AUDIO", "true").lower() == "true"

# ── Ensure dirs exist ──────────────────────────────────────────────────────────
CACHE_DIR.mkdir(parents=True, exist_ok=True)
