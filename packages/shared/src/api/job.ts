// ── Job Queue Contract ─────────────────────────────────────────────────────────
// Shared between Next.js (TypeScript) and Python worker (Pydantic).
// Any change here must be mirrored in services/worker/models.py.

export type JobStatus = "queued" | "processing" | "completed" | "failed";

export type PipelineStep =
  | "download"
  | "separate"
  | "transcribe"
  | "align"
  | "done";

export interface JobStepInfo {
  name: PipelineStep;
  label: string;
  status: JobStatus;
  progress: number;
  error: string | null;
}

export interface Job {
  id: string;
  songId: string;
  youtubeId: string;
  youtubeUrl: string;
  status: JobStatus;
  /** 0–100 overall */
  progress: number;
  currentStep: PipelineStep | null;
  steps: JobStepInfo[];
  error: string | null;
  createdAt: string;
  updatedAt: string;
}

// ── Request / Response shapes ──────────────────────────────────────────────────

export interface CreateJobRequest {
  songId: string;
  youtubeId: string;
  youtubeUrl: string;
}

export interface CreateJobResponse {
  jobId: string;
  status: JobStatus;
}

// ── Result payload (returned when status === "completed") ──────────────────────

export interface LyricWord {
  word: string;
  startTime: number;
  endTime: number;
  score: number;
}

export interface TranscribedLine {
  index: number;
  startTime: number;
  endTime: number;
  text: string;
  words: LyricWord[];
}

export interface JobResult {
  jobId: string;
  songId: string;
  youtubeId: string;
  lyrics: TranscribedLine[];
  metadata: {
    duration: number;
    language: string;
    whisperModel: string;
    vocalsOnly: boolean;
  };
}

// ── Worker health ──────────────────────────────────────────────────────────────

export interface WorkerHealth {
  status: "ok" | "degraded";
  version: string;
  gpuAvailable: boolean;
  whisperModel: string;
  queueDepth: number;
}
