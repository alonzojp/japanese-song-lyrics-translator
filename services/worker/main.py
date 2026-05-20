"""
Japanese Lyrics Worker — FastAPI service.
Handles job creation, status polling, and result retrieval.
Heavy processing runs in a thread pool (one job at a time by default).
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import CACHE_DIR, WHISPER_MODEL, WORKER_CONCURRENCY
from models import CreateJobRequest, JobStatus, WorkerHealth
from pipeline import run_job
from queue import create_job, get_job, get_queue_depth, get_result_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Single global executor limits concurrent heavy jobs
_executor = ThreadPoolExecutor(max_workers=WORKER_CONCURRENCY)

app = FastAPI(
    title="Japanese Lyrics Worker",
    version="0.1.0",
    description="Audio processing pipeline for Japanese song lyrics",
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
    # Dispatch to thread pool (non-blocking)
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
    return job.to_api()


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


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from config import HOST, PORT

    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
