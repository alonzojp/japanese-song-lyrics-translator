"""
Pydantic models — mirrors packages/shared/src/api/job.ts exactly.
Any change here must be reflected in the TypeScript file too.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    queued     = "queued"
    processing = "processing"
    completed  = "completed"
    failed     = "failed"


PipelineStep = Literal["download", "separate", "transcribe", "align", "done"]


# ── Job models ─────────────────────────────────────────────────────────────────

class JobStepInfo(BaseModel):
    name:     PipelineStep
    label:    str
    status:   JobStatus
    progress: int = Field(ge=0, le=100)
    error:    Optional[str] = None
    message:  Optional[str] = None


class Job(BaseModel):
    id:           str
    song_id:      str
    youtube_id:   str
    youtube_url:  str
    status:       JobStatus
    progress:     int = Field(ge=0, le=100)
    current_step: Optional[PipelineStep] = None
    steps:        list[JobStepInfo]
    error:        Optional[str] = None
    recent_logs:  list["LogEntry"] = Field(default_factory=list)
    created_at:   datetime
    updated_at:   datetime

    model_config = {"populate_by_name": True}

    def to_api(self, logs: list[dict] | None = None) -> dict:
        """Serialize to camelCase for the REST API (matches TypeScript contract)."""
        return {
            "id":          self.id,
            "songId":      self.song_id,
            "youtubeId":   self.youtube_id,
            "youtubeUrl":  self.youtube_url,
            "status":      self.status.value,
            "progress":    self.progress,
            "currentStep": self.current_step,
            "steps": [
                {
                    "name":     s.name,
                    "label":    s.label,
                    "status":   s.status.value,
                    "progress": s.progress,
                    "error":    s.error,
                    "message":  s.message,
                }
                for s in self.steps
            ],
            "error":      self.error,
            "recentLogs": logs or [],
            "createdAt":  self.created_at.isoformat(),
            "updatedAt":  self.updated_at.isoformat(),
        }


# ── Request / Response ─────────────────────────────────────────────────────────

class CreateJobRequest(BaseModel):
    song_id:     str = Field(alias="songId")
    youtube_id:  str = Field(alias="youtubeId")
    youtube_url: str = Field(alias="youtubeUrl")

    model_config = {"populate_by_name": True}


class CreateJobResponse(BaseModel):
    job_id: str = Field(alias="jobId")
    status: JobStatus

    model_config = {"populate_by_name": True}


# ── Result payload ─────────────────────────────────────────────────────────────

class LyricWord(BaseModel):
    word:       str
    start_time: float = Field(alias="startTime")
    end_time:   float = Field(alias="endTime")
    score:      float = 1.0

    model_config = {"populate_by_name": True}


class TranscribedLine(BaseModel):
    index:      int
    start_time: float = Field(alias="startTime")
    end_time:   float = Field(alias="endTime")
    text:       str
    words:      list[LyricWord] = []

    model_config = {"populate_by_name": True}


class ResultMetadata(BaseModel):
    duration:      float
    language:      str
    whisper_model: str = Field(alias="whisperModel")
    vocals_only:   bool = Field(alias="vocalsOnly")

    model_config = {"populate_by_name": True}


class JobResult(BaseModel):
    job_id:     str = Field(alias="jobId")
    song_id:    str = Field(alias="songId")
    youtube_id: str = Field(alias="youtubeId")
    lyrics:     list[TranscribedLine]
    metadata:   ResultMetadata

    model_config = {"populate_by_name": True}


# ── Log entry ─────────────────────────────────────────────────────────────────

class LogEntry(BaseModel):
    ts:      str
    level:   Literal["info", "warning", "error"]
    stage:   Optional[str] = None
    message: str


# ── Health ─────────────────────────────────────────────────────────────────────

class WorkerHealth(BaseModel):
    status:        Literal["ok", "degraded"]
    version:       str = "0.1.0"
    gpu_available: bool = Field(alias="gpuAvailable")
    whisper_model: str  = Field(alias="whisperModel")
    queue_depth:   int  = Field(alias="queueDepth")

    model_config = {"populate_by_name": True}
