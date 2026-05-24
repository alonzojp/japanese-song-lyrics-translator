"""
Japanese Lyrics Worker — FastAPI service.
Heavy processing runs in a thread pool (one job at a time by default).
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

from alignment.matcher   import match_lyric_lines, run_offline_match
from cache import cache_dir, cleanup_old, list_cached
from config import CACHE_MAX_AGE_DAYS, WHISPER_MODEL, WORKER_CONCURRENCY
from logs import get_recent_logs, get_best_logs
from lyrics.fetcher import fetch_lyrics, get_cached as get_cached_lyrics
from lyrics.providers.manual import store_manual
from lyrics.types import VideoInfo
from models import CreateJobRequest, JobStatus, WorkerHealth
from pipeline import run_job
from job_queue import create_job, get_job, get_queue_depth, get_result_path, reset_interrupted_jobs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=WORKER_CONCURRENCY)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Startup: reset any jobs interrupted by a previous container crash
    reset_count = reset_interrupted_jobs()
    if reset_count:
        logger.info("Startup: reset %d interrupted job(s) to failed", reset_count)
    # Evict stale cache entries
    deleted = cleanup_old(max_age_days=CACHE_MAX_AGE_DAYS)
    if deleted:
        logger.info("Startup cache cleanup: removed %d stale entries", len(deleted))
    yield
    _executor.shutdown(wait=False)


app = FastAPI(
    title="Japanese Lyrics Worker",
    version="0.1.0",
    description="Audio processing pipeline for Japanese song lyrics",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    try:
        import torch
        gpu = torch.cuda.is_available()
    except ImportError:
        gpu = False

    return WorkerHealth(
        status="ok",
        gpuAvailable=gpu,
        whisperModel=WHISPER_MODEL,
        queueDepth=get_queue_depth(),
    ).model_dump(by_alias=True)


# ── Cache info ─────────────────────────────────────────────────────────────────

@app.get("/cache")
async def cache_info() -> list[dict]:
    """List all cached videos with processing state."""
    return list_cached()


@app.delete("/cache/{youtube_id}")
async def evict_cache(youtube_id: str) -> dict:
    """Force-evict a single video from cache."""
    import shutil
    from config import CACHE_DIR
    target = CACHE_DIR / youtube_id
    if not target.exists():
        raise HTTPException(status_code=404, detail="Not in cache")
    shutil.rmtree(target)
    return {"evicted": youtube_id}


@app.delete("/cache/{youtube_id}/alignment")
async def invalidate_alignment(youtube_id: str) -> dict:
    """
    Invalidate only the transcribe + align stages, preserving audio and vocals.
    Use this when you want WhisperX to re-run (e.g. official lyrics changed).
    """
    from config import CACHE_DIR
    target = CACHE_DIR / youtube_id
    if not target.exists():
        raise HTTPException(status_code=404, detail="Not in cache")

    from cache import invalidate_stage
    invalidate_stage(youtube_id, "transcribe")
    invalidate_stage(youtube_id, "align")

    _files = [
        "transcript.json",
        "aligned_words.json",
        "aligned_lines.json",
        "alignment_meta.json",
        "alignment_selection.json",
        "canonical_lines.json",
        "matched_lyrics.json",
        "alignment_diagnostic.txt",
        "alignment_debug.json",
        "lyrics.json",
    ]
    removed = []
    for name in _files:
        p = target / name
        if p.exists():
            p.unlink()
            removed.append(name)

    return {
        "youtubeId":         youtube_id,
        "invalidatedStages": ["transcribe", "align"],
        "removedFiles":      removed,
    }


@app.delete("/cache/{youtube_id}/postprocess")
async def invalidate_postprocess(youtube_id: str) -> dict:
    """
    Invalidate only the align (post-processing) stage.

    Keeps download, separate, transcribe, and aligned_words.json intact so the
    pipeline skips straight to the Python algorithm layer on the next reprocess.
    Use this when testing changes to timing/alignment algorithms.
    """
    from config import CACHE_DIR
    target = CACHE_DIR / youtube_id
    if not target.exists():
        raise HTTPException(status_code=404, detail="Not in cache")

    from cache import invalidate_stage
    invalidate_stage(youtube_id, "align")

    # Only delete files produced by the Python post-processing layer.
    # aligned_words.json (WhisperX output) is intentionally kept.
    _files = [
        "aligned_lines.json",
        "alignment_meta.json",
        "alignment_selection.json",
        "canonical_lines.json",
        "matched_lyrics.json",
        "alignment_diagnostic.txt",
        "alignment_debug.json",
        "lyrics.json",
    ]
    removed = []
    for name in _files:
        p = target / name
        if p.exists():
            p.unlink()
            removed.append(name)

    return {
        "youtubeId":         youtube_id,
        "invalidatedStages": ["align"],
        "removedFiles":      removed,
    }


# ── Jobs ───────────────────────────────────────────────────────────────────────

@app.post("/jobs", status_code=201)
async def create_job_endpoint(
    body: CreateJobRequest,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    job = create_job(
        song_id=body.song_id,
        youtube_id=body.youtube_id,
        youtube_url=body.youtube_url,
    )
    background_tasks.add_task(_run_in_executor, job.id)
    logger.info("Job %s created for %s", job.id, body.youtube_id)
    return JSONResponse(
        content={"jobId": job.id, "status": job.status.value},
        status_code=201,
    )


def _run_in_executor(job_id: str) -> None:
    _executor.submit(run_job, job_id)


@app.get("/jobs/{job_id}")
async def get_job_endpoint(job_id: str) -> dict:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    # For completed jobs use get_best_logs (richest historical set for the
    # video — includes transcription logs from prior full runs even when the
    # current job was postprocess-only). For in-progress/failed jobs use
    # get_recent_logs so live streaming reflects the current job.
    log_limit = 5000
    if job.status == JobStatus.completed and job.youtube_id:
        logs = get_best_logs(job.youtube_id, limit=log_limit)
    else:
        logs = get_recent_logs(job_id, limit=log_limit)
    return job.to_api(logs=logs)


@app.get("/jobs/{job_id}/result")
async def get_result_endpoint(job_id: str) -> dict:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.completed:
        raise HTTPException(
            status_code=409,
            detail=f"Job is {job.status.value}, not completed",
        )
    result_path = get_result_path(job_id)
    if not result_path or not Path(result_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    data = json.loads(Path(result_path).read_text(encoding="utf-8"))
    data["jobId"]  = job_id
    data["songId"] = job.song_id

    result_dir = Path(result_path).parent

    # Read the canonical alignment written once at job completion.
    # No selector execution, no scoring, no mutation at request time.
    canonical_path = result_dir / "canonical_lines.json"
    if canonical_path.exists():
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
        data["alignedLines"]       = canonical.get("lines", [])
        data["alignmentSelection"] = canonical.get("selectionMeta", {})
        data["matchedStats"]       = {}
    else:
        # Backward compat: job predates canonical_lines.json.
        # Serve acoustic lines directly — already normalized, no mutation needed.
        aligned_path = result_dir / "aligned_lines.json"
        if aligned_path.exists():
            aligned = json.loads(aligned_path.read_text(encoding="utf-8"))
            data["alignedLines"] = aligned.get("lines", [])
        data["alignmentSelection"] = {}
        data["matchedStats"]       = {}

    meta_path = result_dir / "alignment_meta.json"
    if meta_path.exists():
        data["alignmentMeta"] = json.loads(meta_path.read_text(encoding="utf-8"))

    return data


# ── Lyrics ─────────────────────────────────────────────────────────────────────

class FetchLyricsRequest(BaseModel):
    youtube_id:  str = Field(alias="youtubeId")
    youtube_url: str = Field(alias="youtubeUrl")
    title:       Optional[str] = None
    uploader:    Optional[str] = None
    description: Optional[str] = None
    duration:    Optional[float] = None
    force:       bool = False

    model_config = {"populate_by_name": True}


@app.post("/lyrics/fetch")
async def fetch_lyrics_endpoint(body: FetchLyricsRequest) -> dict:
    """
    Fetch lyrics via the provider chain. Returns cached result if available.
    Pass force=true to bypass cache and re-run all providers.
    """
    video = VideoInfo(
        youtube_id=body.youtube_id,
        youtube_url=body.youtube_url,
        title=body.title,
        uploader=body.uploader,
        description=body.description,
        duration=body.duration,
    )
    result = await fetch_lyrics(video, force=body.force)
    return result


@app.get("/cache/{youtube_id}/diagnostic")
async def download_diagnostic(youtube_id: str):
    """Download alignment_diagnostic.txt for a video (plain text, for debugging)."""
    path = cache_dir(youtube_id) / "alignment_diagnostic.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No diagnostic file — run alignment first")
    return FileResponse(str(path), media_type="text/plain",
                        filename=f"{youtube_id}_alignment_diagnostic.txt")


@app.get("/cache/{youtube_id}/logs")
async def get_video_logs(youtube_id: str, limit: int = 2000):
    """
    Return logs from the most log-rich completed job for this video.
    Bypasses currentJobId so re-submitted instant-cache jobs don't hide
    the full diagnostic log from a previous full run.
    """
    logs = get_best_logs(youtube_id, limit=limit)
    logger.info("[log_debug] /cache/%s/logs returning %d entries", youtube_id, len(logs))
    return {"youtubeId": youtube_id, "logCount": len(logs), "logs": logs}


@app.get("/lyrics/{youtube_id}")
async def get_lyrics_endpoint(youtube_id: str) -> dict:
    """Return cached lyrics for a video (404 if not yet fetched)."""
    cached = get_cached_lyrics(youtube_id)
    if not cached:
        raise HTTPException(status_code=404, detail="No cached lyrics for this video")
    return cached


@app.post("/lyrics/{youtube_id}/manual", status_code=201)
async def upload_manual_lyrics(youtube_id: str, body: dict) -> dict:
    """Store user-provided lyrics text as a manual override."""
    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    path = store_manual(youtube_id, text)
    return {"youtubeId": youtube_id, "storedAt": str(path)}


# ── Alignment matching ────────────────────────────────────────────────────────

@app.get("/alignment/{youtube_id}/matched")
async def get_matched_lyrics(youtube_id: str) -> dict:
    """
    Return the offline DTW-matched lyrics for a video.
    Produced by match_lyric_lines(transcript, officialLyrics).
    404 if not yet computed.
    """
    matched_path = cache_dir(youtube_id) / "matched_lyrics.json"
    if not matched_path.exists():
        raise HTTPException(status_code=404, detail="No matched lyrics — run /alignment/{id}/match first")
    return json.loads(matched_path.read_text(encoding="utf-8"))


@app.post("/alignment/{youtube_id}/match")
async def trigger_match(youtube_id: str) -> dict:
    """
    Trigger offline DTW alignment: match official lyrics to transcript.
    Requires both aligned_words.json and lyrics_cached.json to exist.
    """
    job_cache = cache_dir(youtube_id)
    result    = run_offline_match(job_cache, youtube_id)
    if not result:
        raise HTTPException(
            status_code=409,
            detail="Missing aligned_words.json or lyrics_cached.json — process the video and fetch lyrics first",
        )
    return result


@app.post("/alignment/match-inline")
async def match_inline(body: dict) -> dict:
    """
    Run DTW alignment directly on provided transcript + lyrics data.
    Body: { transcript: [{text, startTime, endTime}], officialLines: [{index, text}] }
    """
    transcript     = body.get("transcript", [])
    official_lines = body.get("officialLines", [])
    if not transcript or not official_lines:
        raise HTTPException(status_code=400, detail="transcript and officialLines are required")
    return match_lyric_lines(transcript, official_lines)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from config import HOST, PORT

    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
