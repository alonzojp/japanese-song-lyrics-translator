"""
Step 3 — Transcription + forced alignment via WhisperX.
Input:  vocals.wav (or full audio)
Output: list of segments with word-level timestamps
"""
from __future__ import annotations

import logging
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


def _resolve_device() -> tuple[str, str]:
    """Returns (device, compute_type) based on config + availability."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if WHISPER_DEVICE == "auto":
        device = "cuda" if has_cuda else "cpu"
    else:
        device = WHISPER_DEVICE

    if WHISPER_COMPUTE_TYPE == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    else:
        compute_type = WHISPER_COMPUTE_TYPE

    return device, compute_type


def transcribe_and_align(
    audio_path: Path,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> list[dict]:
    """
    Run WhisperX transcription + alignment.
    Returns list of segment dicts:
      {start, end, text, words: [{word, start, end, score}]}
    """
    import whisperx  # imported lazily — heavy dependency

    device, compute_type = _resolve_device()
    logger.info(
        "Transcribing with model=%s device=%s compute=%s",
        WHISPER_MODEL, device, compute_type,
    )

    if progress_cb:
        progress_cb(5)

    # ── Load model ─────────────────────────────────────────────────────────────
    model = whisperx.load_model(
        WHISPER_MODEL,
        device,
        compute_type=compute_type,
        language=WHISPER_LANGUAGE,
    )

    if progress_cb:
        progress_cb(20)

    # ── Transcribe ─────────────────────────────────────────────────────────────
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(
        audio,
        language=WHISPER_LANGUAGE,
        batch_size=WHISPER_BATCH_SIZE,
    )

    if progress_cb:
        progress_cb(60)

    # Free VRAM before alignment
    del model
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass

    # ── Forced alignment ───────────────────────────────────────────────────────
    align_model, align_metadata = whisperx.load_align_model(
        language_code=WHISPER_LANGUAGE,
        device=device,
    )
    aligned = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    if progress_cb:
        progress_cb(100)

    return aligned.get("segments", [])
