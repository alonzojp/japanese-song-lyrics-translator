"""
Job pipeline — orchestrates the four processing steps and updates queue state.
Runs in a background thread (one job at a time by default).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import soundfile as sf

from config import CACHE_DIR, SKIP_VOCAL_SEPARATION, WHISPER_MODEL
from models import JobStatus
from queue import (
    get_job,
    set_job_status,
    update_job_step,
)
from steps.download import download_audio
from steps.separate import separate_vocals
from steps.transcribe import transcribe_and_align
from steps.align import process_segments

logger = logging.getLogger(__name__)

# Step index → cumulative overall-progress weight
_STEP_WEIGHTS = {
    "download":   (0,  20),   # 0–20 %
    "separate":   (20, 45),   # 20–45 %
    "transcribe": (45, 85),   # 45–85 %
    "align":      (85, 100),  # 85–100 %
}


def _overall(step: str, step_pct: int) -> int:
    lo, hi = _STEP_WEIGHTS[step]
    return lo + int((hi - lo) * step_pct / 100)


def run_job(job_id: str) -> None:
    """Entry point — called in a background thread."""
    job = get_job(job_id)
    if not job:
        logger.error("Job %s not found", job_id)
        return

    youtube_id = job.youtube_id
    cache_dir  = CACHE_DIR / youtube_id

    set_job_status(job_id, JobStatus.processing)
    logger.info("Starting job %s for %s", job_id, youtube_id)

    try:
        # ── Step 1: Download ───────────────────────────────────────────────────
        def dl_progress(pct: int) -> None:
            update_job_step(job_id, "download", JobStatus.processing, pct, _overall("download", pct))

        update_job_step(job_id, "download", JobStatus.processing, 0, 0)
        audio_path, _info = download_audio(job.youtube_url, cache_dir, dl_progress)
        update_job_step(job_id, "download", JobStatus.completed, 100, _STEP_WEIGHTS["download"][1])
        logger.info("[%s] Download complete: %s", job_id, audio_path)

        # ── Step 2: Separate vocals ────────────────────────────────────────────
        def sep_progress(pct: int) -> None:
            update_job_step(job_id, "separate", JobStatus.processing, pct, _overall("separate", pct))

        update_job_step(job_id, "separate", JobStatus.processing, 0, _STEP_WEIGHTS["separate"][0])
        vocals_path = separate_vocals(audio_path, cache_dir, sep_progress)
        update_job_step(job_id, "separate", JobStatus.completed, 100, _STEP_WEIGHTS["separate"][1])
        vocals_only = not SKIP_VOCAL_SEPARATION
        logger.info("[%s] Separation complete: %s", job_id, vocals_path)

        # ── Step 3: Transcribe + align ─────────────────────────────────────────
        def tx_progress(pct: int) -> None:
            update_job_step(job_id, "transcribe", JobStatus.processing, pct, _overall("transcribe", pct))

        update_job_step(job_id, "transcribe", JobStatus.processing, 0, _STEP_WEIGHTS["transcribe"][0])
        raw_segments = transcribe_and_align(vocals_path, tx_progress)
        update_job_step(job_id, "transcribe", JobStatus.completed, 100, _STEP_WEIGHTS["transcribe"][1])
        logger.info("[%s] Transcription complete: %d segments", job_id, len(raw_segments))

        # ── Step 4: Post-process + write lyrics.json ───────────────────────────
        def al_progress(pct: int) -> None:
            update_job_step(job_id, "align", JobStatus.processing, pct, _overall("align", pct))

        update_job_step(job_id, "align", JobStatus.processing, 0, _STEP_WEIGHTS["align"][0])

        try:
            audio_info = sf.info(str(vocals_path))
            duration = audio_info.duration
        except Exception:
            duration = 0.0

        process_segments(
            raw_segments,
            output_dir=cache_dir,
            youtube_id=youtube_id,
            audio_duration=duration,
            vocals_only=vocals_only,
            whisper_model=WHISPER_MODEL,
            progress_cb=al_progress,
        )
        update_job_step(job_id, "align", JobStatus.completed, 100, 100)

        # Mark done
        result_path = str(cache_dir / "lyrics.json")
        set_job_status(job_id, JobStatus.completed, result_path=result_path)
        logger.info("[%s] Job complete → %s", job_id, result_path)

    except Exception as exc:
        logger.exception("[%s] Job failed: %s", job_id, exc)
        set_job_status(job_id, JobStatus.failed, error=str(exc))
