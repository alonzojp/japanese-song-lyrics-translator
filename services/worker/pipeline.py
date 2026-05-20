"""
Job pipeline — orchestrates the four processing steps.

Cache-aware: before each step it checks the manifest.
If a stage is already marked complete and its output file exists, it is skipped.
This means re-submitting the same YouTube video is instant (all cache hits).

Stage    Weight    Overall %
-------  --------  ---------
download   0–25
separate  25–55
transcribe 55–88
align     88–100
"""
from __future__ import annotations

import logging
from pathlib import Path

import soundfile as sf

from cache import (
    all_stages_complete,
    cache_dir,
    cleanup_temp_files,
    get_stage_data,
    is_stage_complete,
    store_metadata,
)
from config import CACHE_DIR, SKIP_VOCAL_SEPARATION, WHISPER_MODEL
from logs import JobLogger
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

_WEIGHTS = {
    "download":   (0,  25),
    "separate":   (25, 55),
    "transcribe": (55, 88),
    "align":      (88, 100),
}


def _overall(step: str, pct: int) -> int:
    lo, hi = _WEIGHTS[step]
    return lo + int((hi - lo) * pct / 100)


def run_job(job_id: str) -> None:
    job = get_job(job_id)
    if not job:
        logger.error("Job %s not found", job_id)
        return

    jl        = JobLogger(job_id)
    yt_id     = job.youtube_id
    job_cache = cache_dir(yt_id)

    set_job_status(job_id, JobStatus.processing)
    jl.info(f"Starting job {job_id[:8]}… for youtube_id={yt_id}")

    # If the entire pipeline was already run for this video, finish immediately
    if all_stages_complete(yt_id):
        jl.info("All stages already cached — instant completion")
        _mark_all_steps_complete(job_id)
        result_path = str(job_cache / "lyrics.json")
        set_job_status(job_id, JobStatus.completed, result_path=result_path)
        return

    try:
        # ── Step 1: Download ───────────────────────────────────────────────────
        def dl_cb(pct: int) -> None:
            update_job_step(job_id, "download", JobStatus.processing, pct, _overall("download", pct))

        cached_dl = is_stage_complete(yt_id, "download")
        status_dl = JobStatus.completed if cached_dl else JobStatus.processing
        update_job_step(job_id, "download", status_dl, 0 if not cached_dl else 100, _WEIGHTS["download"][0])

        audio_path, metadata = download_audio(
            youtube_url=job.youtube_url,
            youtube_id=yt_id,
            output_dir=job_cache,
            progress_cb=dl_cb,
            job_log=jl,
        )

        # Store metadata in cache manifest (title, uploader, duration, thumbnail)
        if metadata:
            store_metadata(yt_id, metadata)

        update_job_step(job_id, "download", JobStatus.completed, 100, _WEIGHTS["download"][1])

        # ── Step 2: Separate vocals ────────────────────────────────────────────
        def sep_cb(pct: int) -> None:
            update_job_step(job_id, "separate", JobStatus.processing, pct, _overall("separate", pct))

        cached_sep = is_stage_complete(yt_id, "separate") or SKIP_VOCAL_SEPARATION
        update_job_step(
            job_id, "separate",
            JobStatus.completed if cached_sep else JobStatus.processing,
            100 if cached_sep else 0,
            _WEIGHTS["separate"][0],
        )

        vocals_path = separate_vocals(
            audio_path=audio_path,
            output_dir=job_cache,
            youtube_id=yt_id,
            progress_cb=sep_cb,
            job_log=jl,
        )
        vocals_only = not SKIP_VOCAL_SEPARATION

        update_job_step(job_id, "separate", JobStatus.completed, 100, _WEIGHTS["separate"][1])

        # ── Step 3: Transcribe + align ─────────────────────────────────────────
        def tx_cb(pct: int) -> None:
            update_job_step(job_id, "transcribe", JobStatus.processing, pct, _overall("transcribe", pct))

        cached_tx = is_stage_complete(yt_id, "transcribe")
        update_job_step(
            job_id, "transcribe",
            JobStatus.completed if cached_tx else JobStatus.processing,
            100 if cached_tx else 0,
            _WEIGHTS["transcribe"][0],
        )

        raw_segments = transcribe_and_align(
            audio_path=vocals_path,
            output_dir=job_cache,
            youtube_id=yt_id,
            progress_cb=tx_cb,
            job_log=jl,
        )

        update_job_step(job_id, "transcribe", JobStatus.completed, 100, _WEIGHTS["transcribe"][1])

        # ── Step 4: Post-process ───────────────────────────────────────────────
        def al_cb(pct: int) -> None:
            update_job_step(job_id, "align", JobStatus.processing, pct, _overall("align", pct))

        cached_al = is_stage_complete(yt_id, "align")
        update_job_step(
            job_id, "align",
            JobStatus.completed if cached_al else JobStatus.processing,
            100 if cached_al else 0,
            _WEIGHTS["align"][0],
        )

        try:
            duration = sf.info(str(vocals_path)).duration
        except Exception:
            duration = metadata.get("duration", 0.0) if metadata else 0.0

        process_segments(
            raw_segments=raw_segments,
            output_dir=job_cache,
            youtube_id=yt_id,
            audio_duration=duration,
            vocals_only=vocals_only,
            whisper_model=WHISPER_MODEL,
            progress_cb=al_cb,
            job_log=jl,
        )

        update_job_step(job_id, "align", JobStatus.completed, 100, 100)

        # ── Cleanup temps ──────────────────────────────────────────────────────
        cleanup_temp_files(yt_id)

        result_path = str(job_cache / "lyrics.json")
        set_job_status(job_id, JobStatus.completed, result_path=result_path)
        jl.info(f"Job complete → {result_path}")

    except Exception as exc:
        jl.error(f"Job failed: {exc}", stage=None)
        logger.exception("[%s] Job failed", job_id)
        set_job_status(job_id, JobStatus.failed, error=str(exc))


def _mark_all_steps_complete(job_id: str) -> None:
    """Mark all steps as completed instantly (full cache hit path)."""
    for step, (_, hi) in _WEIGHTS.items():
        update_job_step(job_id, step, JobStatus.completed, 100, hi)
