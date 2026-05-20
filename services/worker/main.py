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
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

from cache import cleanup_old, list_cached
from config import CACHE_MAX_AGE_DAYS, WHISPER_MODEL, WORKER_CONCURRENCY
from logs import get_recent_logs
from lyrics.fetcher import fetch_lyrics, get_cached as get_cached_lyrics
from lyrics.providers.manual import store_manual
from lyrics.types import VideoInfo
from models import CreateJobRequest, JobStatus, WorkerHealth
from pipeline import run_job
from queue import create_job, get_job, get_queue_depth, get_result_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=WORKER_CONCURRENCY)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Startup: evict stale cache entries
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
    logs = get_recent_logs(job_id, limit=40)
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


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from config import HOST, PORT

    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
