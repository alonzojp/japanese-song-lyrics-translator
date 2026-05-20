"""
Job pipeline — orchestrates the four processing steps.

Cache-aware: each step checks its manifest entry before running.
Re-submitting the same YouTube video is instant when all stages are cached.

Stage weights (overall progress %):
  download    0–20
  separate   20–45
  transcribe 45–85
  align      85–100
"""
from __future__ import annotations

import logging
from pathlib import Path

from cache import (
    all_stages_complete,
    cache_dir,
    cleanup_temp_files,
    is_stage_complete,
    store_metadata,
)
from config import CACHE_DIR, SKIP_VOCAL_SEPARATION, WHISPER_MODEL
from logs import JobLogger
from models import JobStatus
from queue import get_job, set_job_status, update_job_step
from alignment.matcher import run_offline_match
from steps.download import download_audio
from steps.separate import separate_vocals
from steps.transcribe import run_transcription
from steps.align import run_alignment_postprocess

logger = logging.getLogger(__name__)

_WEIGHTS = {
    "download":   (0,  20),
    "separate":   (20, 45),
    "transcribe": (45, 85),
    "align":      (85, 100),
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
    jl.info(f"Starting job {job_id[:8]}… youtube_id={yt_id}")

    if all_stages_complete(yt_id):
        jl.info("All stages cached — instant completion")
        _mark_all_complete(job_id)
        set_job_status(job_id, JobStatus.completed,
                       result_path=str(job_cache / "lyrics.json"))
        return

    try:
        # ── Step 1: Download ───────────────────────────────────────────────────
        def dl_cb(pct: int) -> None:
            update_job_step(job_id, "download", JobStatus.processing, pct, _overall("download", pct))

        cached_dl = is_stage_complete(yt_id, "download")
        update_job_step(job_id, "download",
                        JobStatus.completed if cached_dl else JobStatus.processing,
                        100 if cached_dl else 0, _WEIGHTS["download"][0])

        audio_path, metadata = download_audio(
            youtube_url=job.youtube_url,
            youtube_id=yt_id,
            output_dir=job_cache,
            progress_cb=dl_cb,
            job_log=jl,
        )
        if metadata:
            store_metadata(yt_id, metadata)
        update_job_step(job_id, "download", JobStatus.completed, 100, _WEIGHTS["download"][1])

        # ── Step 2: Separate vocals ────────────────────────────────────────────
        def sep_cb(pct: int) -> None:
            update_job_step(job_id, "separate", JobStatus.processing, pct, _overall("separate", pct))

        cached_sep = is_stage_complete(yt_id, "separate") or SKIP_VOCAL_SEPARATION
        update_job_step(job_id, "separate",
                        JobStatus.completed if cached_sep else JobStatus.processing,
                        100 if cached_sep else 0, _WEIGHTS["separate"][0])

        vocals_path = separate_vocals(
            audio_path=audio_path,
            output_dir=job_cache,
            youtube_id=yt_id,
            progress_cb=sep_cb,
            job_log=jl,
        )
        vocals_only = not SKIP_VOCAL_SEPARATION
        update_job_step(job_id, "separate", JobStatus.completed, 100, _WEIGHTS["separate"][1])

        # ── Step 3: Transcription + forced alignment ───────────────────────────
        def tx_cb(pct: int) -> None:
            update_job_step(job_id, "transcribe", JobStatus.processing, pct, _overall("transcribe", pct))

        cached_tx = is_stage_complete(yt_id, "transcribe")
        update_job_step(job_id, "transcribe",
                        JobStatus.completed if cached_tx else JobStatus.processing,
                        100 if cached_tx else 0, _WEIGHTS["transcribe"][0])

        aligned_segments, backend, align_method = run_transcription(
            audio_path=vocals_path,
            output_dir=job_cache,
            youtube_id=yt_id,
            progress_cb=tx_cb,
            job_log=jl,
        )
        update_job_step(job_id, "transcribe", JobStatus.completed, 100, _WEIGHTS["transcribe"][1])

        # ── Step 4: Post-process → LyricLine files ─────────────────────────────
        def al_cb(pct: int) -> None:
            update_job_step(job_id, "align", JobStatus.processing, pct, _overall("align", pct))

        cached_al = is_stage_complete(yt_id, "align")
        update_job_step(job_id, "align",
                        JobStatus.completed if cached_al else JobStatus.processing,
                        100 if cached_al else 0, _WEIGHTS["align"][0])

        _lines, confidence = run_alignment_postprocess(
            aligned_segments=aligned_segments,
            output_dir=job_cache,
            youtube_id=yt_id,
            audio_path=vocals_path,
            backend=backend,
            method=align_method,
            vocals_only=vocals_only,
            progress_cb=al_cb,
            job_log=jl,
        )
        update_job_step(job_id, "align", JobStatus.completed, 100, 100)

        cleanup_temp_files(yt_id)

        # ── Optional step 5: offline DTW match against official lyrics ─────────
        # Runs only if lyrics_cached.json exists (from the lyrics provider system).
        try:
            match_result = run_offline_match(job_cache, yt_id, job_log=jl)
            if match_result:
                jl.info(
                    f"Offline match: {len(match_result['lines'])} lines, "
                    f"avg_conf={match_result['stats'].get('avgConfidence', 0):.2f}",
                )
        except Exception as match_exc:
            jl.warning(f"Offline match skipped: {match_exc}")

        result_path = str(job_cache / "lyrics.json")
        set_job_status(job_id, JobStatus.completed, result_path=result_path)
        jl.info(
            f"Job complete → {len(_lines)} lines, "
            f"confidence={confidence.get('overall', 0):.2f}, "
            f"backend={backend}, method={align_method}"
        )

    except Exception as exc:
        jl.error(f"Job failed: {exc}")
        logger.exception("[%s] Job failed", job_id)
        set_job_status(job_id, JobStatus.failed, error=str(exc))


def _mark_all_complete(job_id: str) -> None:
    for step, (_, hi) in _WEIGHTS.items():
        update_job_step(job_id, step, JobStatus.completed, 100, hi)
