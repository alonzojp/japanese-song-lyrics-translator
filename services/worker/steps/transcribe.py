"""
Step 3 — Transcription + forced alignment.

Sub-steps (all cached independently):
  3a. audio_prep   → vocals_asr.wav          (ffmpeg: 16kHz mono, filtered)
  3b. transcribe   → transcript.json         (Whisper ASR via best backend)
  3c. force_align  → aligned_words.json      (WhisperX / aeneas / even)

Cache behaviour:
  - Each sub-step checks for its output file; re-runs only missing stages.
  - Full cache hit → skips all sub-steps, loads and returns stored segments.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import soundfile as sf

from alignment import (
    force_align,
    load_aligned_words,
    load_transcript,
    prepare_audio_for_asr,
    save_aligned_words,
    save_transcript,
    transcribe,
)
from cache import is_stage_complete, mark_stage_complete

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[int], None]]


def run_transcription(
    audio_path:      Path,
    output_dir:      Path,
    youtube_id:      str,
    official_lyrics: Optional[list[dict]] = None,
    progress_cb:     ProgressCallback = None,
    message_cb=None,
    job_log=None,
) -> tuple[list[dict], str, str]:
    """
    Full transcription + forced alignment for one audio file.

    Returns (enriched_segments, transcription_backend, alignment_method).

    Progress: 0 → 100 across all three sub-steps.
    """
    def log(msg: str) -> None:
        if job_log:
            job_log.info(msg, stage="transcribe")
        logger.info(msg)

    def sub_cb(fraction_of_total: float) -> Callable[[int], None]:
        """Wrap a sub-step progress into an overall range."""
        def _cb(pct: int) -> None:
            if progress_cb:
                progress_cb(int(fraction_of_total * pct))
        return _cb

    # ── Full cache hit ─────────────────────────────────────────────────────────
    # load_aligned_words performs a version check — returns None when the cached
    # file was written by an older aligner version, forcing a fresh alignment run.
    if is_stage_complete(youtube_id, "transcribe"):
        loaded = load_aligned_words(output_dir)
        if loaded:
            segs, method = loaded
            log(f"Cache hit — loaded {len(segs)} aligned segments for {youtube_id}")
            if progress_cb:
                progress_cb(100)
            return segs, "cached", method
        # Version mismatch: fall through to re-run alignment
        log("Aligner version mismatch — re-running forced alignment")

    # ── 3a: Audio prep (0–20 %) ────────────────────────────────────────────────
    log("Preparing audio for ASR …")
    if message_cb:
        message_cb("Preparing audio for transcription…")
    asr_path = prepare_audio_for_asr(
        audio_path,
        output_dir,
        progress_cb=sub_cb(0.20),
        job_log=job_log,
    )

    # ── 3b: Transcription (20–70 %) ────────────────────────────────────────────
    raw_segments = load_transcript(output_dir)
    if raw_segments is not None:
        log(f"Loaded cached transcript ({len(raw_segments)} segments)")
        backend = "cached"
    else:
        log("Transcribing …")
        if message_cb:
            message_cb("Loading Whisper model…")

        def tx_cb(pct: int) -> None:
            if progress_cb:
                progress_cb(20 + int(0.50 * pct))

        raw_segments, backend = transcribe(asr_path, progress_cb=tx_cb, message_cb=message_cb, job_log=job_log)
        log(f"Transcription complete: {len(raw_segments)} segments via {backend}")
        save_transcript(output_dir, raw_segments, backend)

    # ── 3c: Forced alignment (70–100 %) ────────────────────────────────────────
    loaded_aligned = load_aligned_words(output_dir)

    # If official lyrics are available but previous alignment didn't use them,
    # re-run alignment with official lyrics for much better timing accuracy.
    if loaded_aligned and official_lyrics:
        _, cached_method = loaded_aligned
        if cached_method != "whisperx_official":
            log("Official lyrics available — re-running alignment for precise timestamps")
            loaded_aligned = None  # invalidate cache

    if loaded_aligned:
        aligned_segs, method = loaded_aligned
        log(f"Loaded cached alignment ({method}) from {output_dir / 'aligned_words.json'}")
    else:
        if official_lyrics:
            log(f"Running forced alignment with {len(official_lyrics)} official lyrics lines…")
            if message_cb:
                message_cb(f"Aligning {len(official_lyrics)} official lyrics to audio…")
        else:
            log("Running forced alignment …")
            if message_cb:
                message_cb("Aligning words to audio…")

        def al_cb(pct: int) -> None:
            if progress_cb:
                progress_cb(70 + int(0.30 * pct))

        aligned_segs, method = force_align(
            raw_segments, asr_path,
            official_lyrics=official_lyrics,
            progress_cb=al_cb, job_log=job_log,
        )
        log(f"Alignment complete via {method}: {len(aligned_segs)} segments")
        saved = save_aligned_words(output_dir, aligned_segs, method)
        log(f"Wrote aligned_words.json → {saved}")

    # ── Update manifest ────────────────────────────────────────────────────────
    mark_stage_complete(youtube_id, "transcribe", {
        "transcriptPath":    "transcript.json",
        "alignedWordsPath":  "aligned_words.json",
        "segmentCount":      len(aligned_segs),
        "backend":           backend,
        "alignmentMethod":   method,
    })

    if progress_cb:
        progress_cb(100)
    return aligned_segs, backend, method
