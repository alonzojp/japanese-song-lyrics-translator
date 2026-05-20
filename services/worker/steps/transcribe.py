"""
Step 3 — Transcription + forced alignment via WhisperX.
Input:  vocals.wav (or full audio)
Output: list of segments with word-level timestamps

Cache behaviour: skips if manifest marks transcribe as complete
and the raw transcript JSON file exists.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

from cache import is_stage_complete, mark_stage_complete
from config import (
    WHISPER_BATCH_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL,
)

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[int], None]]
TRANSCRIPT_FILE  = "transcript_raw.json"


def _resolve_device() -> tuple[str, str]:
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    device       = ("cuda" if has_cuda else "cpu") if WHISPER_DEVICE == "auto" else WHISPER_DEVICE
    compute_type = ("float16" if device == "cuda" else "int8") if WHISPER_COMPUTE_TYPE == "auto" else WHISPER_COMPUTE_TYPE
    return device, compute_type


def transcribe_and_align(
    audio_path:  Path,
    output_dir:  Path,
    youtube_id:  str,
    progress_cb: ProgressCallback = None,
    job_log=None,
) -> list[dict]:
    """
    Run WhisperX transcription + forced alignment.
    Returns list of aligned segment dicts:
      {start, end, text, words: [{word, start, end, score}]}
    """
    import whisperx  # lazy — heavy import

    def log(msg: str) -> None:
        if job_log:
            job_log.info(msg, stage="transcribe")
        logger.info(msg)

    transcript_path = output_dir / TRANSCRIPT_FILE

    # ── Cache hit ──────────────────────────────────────────────────────────────
    if is_stage_complete(youtube_id, "transcribe") and transcript_path.exists():
        log(f"Cache hit — loading saved transcript for {youtube_id}")
        segments = json.loads(transcript_path.read_text(encoding="utf-8"))
        if progress_cb:
            progress_cb(100)
        return segments

    device, compute_type = _resolve_device()
    log(
        f"Loading Whisper model={WHISPER_MODEL} device={device} "
        f"compute={compute_type} …"
    )
    if progress_cb:
        progress_cb(5)

    model = whisperx.load_model(
        WHISPER_MODEL,
        device,
        compute_type=compute_type,
        language=WHISPER_LANGUAGE,
    )
    if progress_cb:
        progress_cb(20)

    log(f"Transcribing {audio_path.name} (batch_size={WHISPER_BATCH_SIZE}) …")
    audio  = whisperx.load_audio(str(audio_path))
    result = model.transcribe(
        audio,
        language=WHISPER_LANGUAGE,
        batch_size=WHISPER_BATCH_SIZE,
    )
    seg_count = len(result.get("segments", []))
    log(f"Transcription complete: {seg_count} raw segments")
    if progress_cb:
        progress_cb(60)

    # Free VRAM before alignment pass
    del model
    try:
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
    except Exception:
        pass

    log("Running forced alignment …")
    align_model, align_meta = whisperx.load_align_model(
        language_code=WHISPER_LANGUAGE,
        device=device,
    )
    aligned = whisperx.align(
        result["segments"],
        align_model,
        align_meta,
        audio,
        device,
        return_char_alignments=False,
    )
    segments: list[dict] = aligned.get("segments", [])
    log(f"Alignment complete: {len(segments)} segments with word timestamps")
    if progress_cb:
        progress_cb(95)

    # Persist raw transcript for cache
    transcript_path.write_text(
        json.dumps(segments, ensure_ascii=False),
        encoding="utf-8",
    )
    mark_stage_complete(youtube_id, "transcribe", {
        "transcriptPath": TRANSCRIPT_FILE,
        "segmentCount":   len(segments),
    })

    if progress_cb:
        progress_cb(100)
    return segments
