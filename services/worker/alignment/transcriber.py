"""
Whisper transcription with automatic backend selection and retry.

Backend priority (highest to lowest quality):
  1. WhisperX        — best word-level timestamps, forced alignment
  2. faster-whisper  — fast CTranslate2 backend, word-level via VAD
  3. openai-whisper  — original, segment-level only

All backends produce the same output format:
  list[dict] with keys: start, end, text, words (may be empty)

Retry behaviour on CUDA OOM:
  - Halve batch_size up to 3 times before falling back to CPU
  - If CPU also fails, re-raise
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

from config import (
    WHISPER_BATCH_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL,
)

logger = logging.getLogger(__name__)

MAX_RETRY_ATTEMPTS = 3
MIN_BATCH_SIZE     = 1


def _resolve_device() -> tuple[str, str]:
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    device       = ("cuda" if has_cuda else "cpu") if WHISPER_DEVICE == "auto" else WHISPER_DEVICE
    compute_type = ("float16" if device == "cuda" else "int8") \
                    if WHISPER_COMPUTE_TYPE == "auto" else WHISPER_COMPUTE_TYPE
    return device, compute_type


def _is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda" in msg


# ── WhisperX backend ───────────────────────────────────────────────────────────

def _transcribe_whisperx(
    audio_path: Path,
    progress_cb: Optional[Callable[[int], None]],
    job_log,
) -> list[dict]:
    import whisperx

    device, compute_type = _resolve_device()
    batch_size = WHISPER_BATCH_SIZE

    def log(msg: str) -> None:
        if job_log:
            job_log.info(msg, stage="transcribe")
        logger.info(msg)

    log(f"[WhisperX] model={WHISPER_MODEL} device={device} compute={compute_type}")

    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            model = whisperx.load_model(
                WHISPER_MODEL, device,
                compute_type=compute_type,
                language=WHISPER_LANGUAGE,
            )
            if progress_cb:
                progress_cb(25)

            audio  = whisperx.load_audio(str(audio_path))
            result = model.transcribe(
                audio,
                language=WHISPER_LANGUAGE,
                batch_size=batch_size,
            )
            segments = result.get("segments", [])
            log(f"[WhisperX] transcribed {len(segments)} segments (batch_size={batch_size})")

            if progress_cb:
                progress_cb(70)

            # Free VRAM
            del model
            _free_vram()

            return segments

        except Exception as exc:
            if _is_oom(exc) and attempt < MAX_RETRY_ATTEMPTS - 1:
                old_bs   = batch_size
                batch_size = max(MIN_BATCH_SIZE, batch_size // 2)
                log(f"[WhisperX] CUDA OOM on attempt {attempt+1}, retrying batch_size {old_bs}→{batch_size}")
                _free_vram()
                time.sleep(2)
            elif _is_oom(exc) and device == "cuda":
                log("[WhisperX] Falling back to CPU after repeated OOM")
                device       = "cpu"
                compute_type = "int8"
                batch_size   = WHISPER_BATCH_SIZE
            else:
                raise

    raise RuntimeError("WhisperX transcription failed after all retries")


# ── faster-whisper backend ─────────────────────────────────────────────────────

def _transcribe_faster_whisper(
    audio_path: Path,
    progress_cb: Optional[Callable[[int], None]],
    job_log,
) -> list[dict]:
    from faster_whisper import WhisperModel

    device, compute_type = _resolve_device()

    def log(msg: str) -> None:
        if job_log:
            job_log.info(msg, stage="transcribe")
        logger.info(msg)

    log(f"[faster-whisper] model={WHISPER_MODEL} device={device} compute={compute_type}")

    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
            if progress_cb:
                progress_cb(25)

            segments_gen, _info = model.transcribe(
                str(audio_path),
                language=WHISPER_LANGUAGE,
                word_timestamps=True,
                vad_filter=True,
            )

            segments: list[dict] = []
            for seg in segments_gen:
                words = [
                    {"word": w.word, "start": w.start, "end": w.end, "score": w.probability}
                    for w in (seg.words or [])
                ]
                segments.append({
                    "start": seg.start,
                    "end":   seg.end,
                    "text":  seg.text.strip(),
                    "words": words,
                })

            log(f"[faster-whisper] transcribed {len(segments)} segments")
            if progress_cb:
                progress_cb(70)
            return segments

        except Exception as exc:
            if _is_oom(exc) and attempt < MAX_RETRY_ATTEMPTS - 1:
                log(f"[faster-whisper] OOM on attempt {attempt+1}, retrying on CPU")
                device       = "cpu"
                compute_type = "int8"
                _free_vram()
                time.sleep(2)
            else:
                raise

    raise RuntimeError("faster-whisper transcription failed after all retries")


# ── openai-whisper backend ─────────────────────────────────────────────────────

def _transcribe_openai_whisper(
    audio_path: Path,
    progress_cb: Optional[Callable[[int], None]],
    job_log,
) -> list[dict]:
    import whisper

    device = _resolve_device()[0]

    def log(msg: str) -> None:
        if job_log:
            job_log.info(msg, stage="transcribe")
        logger.info(msg)

    log(f"[openai-whisper] model={WHISPER_MODEL} device={device}")

    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            model  = whisper.load_model(WHISPER_MODEL, device=device)
            if progress_cb:
                progress_cb(25)

            result = model.transcribe(
                str(audio_path),
                language=WHISPER_LANGUAGE,
                word_timestamps=True,
                verbose=False,
            )
            raw_segs = result.get("segments", [])
            segments = []
            for seg in raw_segs:
                words = [
                    {"word": w["word"].strip(), "start": w["start"], "end": w["end"], "score": w.get("probability", 1.0)}
                    for w in seg.get("words", [])
                ]
                segments.append({
                    "start": seg["start"],
                    "end":   seg["end"],
                    "text":  seg["text"].strip(),
                    "words": words,
                })

            log(f"[openai-whisper] transcribed {len(segments)} segments")
            if progress_cb:
                progress_cb(70)
            return segments

        except Exception as exc:
            if _is_oom(exc) and attempt < MAX_RETRY_ATTEMPTS - 1:
                log(f"[openai-whisper] OOM on attempt {attempt+1}, retrying on CPU")
                device = "cpu"
                _free_vram()
                time.sleep(2)
            else:
                raise

    raise RuntimeError("openai-whisper transcription failed after all retries")


# ── Public entry point ─────────────────────────────────────────────────────────

def transcribe(
    audio_path:  Path,
    progress_cb: Optional[Callable[[int], None]] = None,
    job_log=None,
) -> tuple[list[dict], str]:
    """
    Transcribe audio using the best available backend.
    Returns (segments, backend_name).

    Each segment: {start, end, text, words: [{word, start, end, score}]}
    """
    backends = [
        ("whisperx",        _transcribe_whisperx),
        ("faster-whisper",  _transcribe_faster_whisper),
        ("openai-whisper",  _transcribe_openai_whisper),
    ]

    last_exc: Optional[Exception] = None

    for name, fn in backends:
        try:
            segments = fn(audio_path, progress_cb, job_log)
            return segments, name
        except ImportError:
            if job_log:
                job_log.info(f"{name} not installed, trying next backend", stage="transcribe")
        except Exception as exc:
            last_exc = exc
            if job_log:
                job_log.warning(f"{name} failed: {exc}", stage="transcribe")

    raise RuntimeError(
        f"All transcription backends failed. Last error: {last_exc}"
    )


def _free_vram() -> None:
    try:
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
