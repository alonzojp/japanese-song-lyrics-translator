import type {
  CreateJobRequest,
  CreateJobResponse,
  FetchLyricsRequest,
  Job,
  JobLogEntry,
  JobResult,
  LyricsResult,
  WorkerHealth,
} from "@japanese-lyrics/shared";

const WORKER_URL = process.env.WORKER_URL ?? "http://localhost:8000";

async function workerFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${WORKER_URL}${path}`, {
    ...init,
    cache: "no-store",   // never cache — job status must always be fresh
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`Worker ${path} → ${res.status}: ${text}`);
  }

  return res.json() as Promise<T>;
}

export const workerClient = {
  // ── Health ──────────────────────────────────────────────────────────────────
  health(): Promise<WorkerHealth> {
    return workerFetch("/health");
  },

  // ── Jobs ────────────────────────────────────────────────────────────────────
  createJob(body: CreateJobRequest): Promise<CreateJobResponse> {
    return workerFetch("/jobs", { method: "POST", body: JSON.stringify(body) });
  },

  getJob(jobId: string): Promise<Job> {
    return workerFetch(`/jobs/${jobId}`);
  },

  getResult(jobId: string): Promise<JobResult> {
    return workerFetch(`/jobs/${jobId}/result`);
  },

  // ── Lyrics ──────────────────────────────────────────────────────────────────
  fetchLyrics(body: FetchLyricsRequest): Promise<LyricsResult> {
    return workerFetch("/lyrics/fetch", { method: "POST", body: JSON.stringify(body) });
  },

  getLyrics(youtubeId: string): Promise<LyricsResult> {
    return workerFetch(`/lyrics/${youtubeId}`);
  },

  uploadManualLyrics(youtubeId: string, text: string): Promise<{ youtubeId: string; storedAt: string }> {
    return workerFetch(`/lyrics/${youtubeId}/manual`, {
      method: "POST",
      body: JSON.stringify({ text }),
    });
  },

  matchLyrics(youtubeId: string): Promise<{ lines: Array<{ text: string; startTime: number; endTime: number; words?: unknown[] }> }> {
    return workerFetch(`/alignment/${youtubeId}/match`, { method: "POST" });
  },

  getBestLogs(youtubeId: string): Promise<{ youtubeId: string; logCount: number; logs: JobLogEntry[] }> {
    return workerFetch(`/cache/${youtubeId}/logs`);
  },

  invalidateForceAlign(youtubeId: string): Promise<{ youtubeId: string; invalidatedStages: string[]; removedFiles: string[] }> {
    return workerFetch(`/cache/${youtubeId}/force-align`, { method: "DELETE" });
  },
};
