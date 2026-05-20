import type {
  CreateJobRequest,
  CreateJobResponse,
  Job,
  JobResult,
  WorkerHealth,
} from "@japanese-lyrics/shared";

const WORKER_URL = process.env.WORKER_URL ?? "http://localhost:8000";

async function workerFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${WORKER_URL}${path}`, {
    ...init,
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
  health(): Promise<WorkerHealth> {
    return workerFetch("/health");
  },

  createJob(body: CreateJobRequest): Promise<CreateJobResponse> {
    return workerFetch("/jobs", {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  getJob(jobId: string): Promise<Job> {
    return workerFetch(`/jobs/${jobId}`);
  },

  getResult(jobId: string): Promise<JobResult> {
    return workerFetch(`/jobs/${jobId}/result`);
  },
};
