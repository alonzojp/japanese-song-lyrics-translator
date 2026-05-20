"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Play, Pause, Loader2, CheckCircle2, XCircle, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Job, JobStepInfo, JobStatus, LyricLine } from "@japanese-lyrics/shared";

// ── Types ──────────────────────────────────────────────────────────────────────

interface KaraokePlayerProps {
  songId: string;
  youtubeId: string;
  title: string | null;
  artist: string | null;
  initialJobId: string | null;
  initialStatus: string | null;
  initialLyrics: LyricLine[];
}

const STEP_LABELS: Record<string, string> = {
  download:   "Downloading audio",
  separate:   "Separating vocals",
  transcribe: "Transcribing speech",
  align:      "Aligning timestamps",
  done:       "Complete",
};

const POLL_INTERVAL_MS = 2500;

// ── Step progress indicator ────────────────────────────────────────────────────

function StepRow({ step }: { step: JobStepInfo }) {
  const icon =
    step.status === "completed" ? (
      <CheckCircle2 className="h-4 w-4 text-green-500" />
    ) : step.status === "processing" ? (
      <Loader2 className="h-4 w-4 animate-spin text-primary" />
    ) : step.status === "failed" ? (
      <XCircle className="h-4 w-4 text-destructive" />
    ) : (
      <div className="h-4 w-4 rounded-full border border-muted-foreground/30" />
    );

  return (
    <div className="flex items-center gap-3">
      {icon}
      <div className="flex-1">
        <div className="flex items-center justify-between text-sm">
          <span
            className={
              step.status === "processing"
                ? "text-foreground font-medium"
                : step.status === "completed"
                ? "text-muted-foreground"
                : "text-muted-foreground/50"
            }
          >
            {step.label}
          </span>
          {step.status === "processing" && (
            <span className="text-xs text-muted-foreground">{step.progress}%</span>
          )}
        </div>
        {step.status === "processing" && (
          <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-primary transition-all duration-500"
              style={{ width: `${step.progress}%` }}
            />
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export function KaraokePlayer({
  songId,
  youtubeId,
  title,
  artist,
  initialJobId,
  initialStatus,
  initialLyrics,
}: KaraokePlayerProps) {
  const [jobId, setJobId]         = useState<string | null>(initialJobId);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(
    initialStatus as JobStatus | null
  );
  const [job, setJob]             = useState<Job | null>(null);
  const [lyrics, setLyrics]       = useState<LyricLine[]>(initialLyrics);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime]             = useState(0);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── Polling ──────────────────────────────────────────────────────────────────

  const poll = useCallback(async (id: string) => {
    try {
      const res = await fetch(`/api/jobs/${id}`);
      if (!res.ok) return;
      const data = (await res.json()) as Job;
      setJob(data);
      setJobStatus(data.status);

      if (data.status === "completed") {
        // Reload lyrics from DB (the GET /api/jobs/[id] route stores them)
        const songRes = await fetch(`/api/songs/${songId}/lyrics`);
        if (songRes.ok) {
          const loaded = (await songRes.json()) as LyricLine[];
          setLyrics(loaded);
        }
      } else if (data.status === "processing" || data.status === "queued") {
        pollRef.current = setTimeout(() => poll(id), POLL_INTERVAL_MS);
      }
    } catch {
      // retry on next interval
      pollRef.current = setTimeout(() => poll(id), POLL_INTERVAL_MS);
    }
  }, [songId]);

  useEffect(() => {
    if (
      jobId &&
      (jobStatus === "processing" || jobStatus === "queued")
    ) {
      poll(jobId);
    }
    return () => {
      if (pollRef.current) clearTimeout(pollRef.current);
    };
  }, [jobId, jobStatus, poll]);

  // ── Submit job ───────────────────────────────────────────────────────────────

  async function submitJob() {
    setSubmitError(null);
    try {
      const res = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ songId }),
      });
      const data = (await res.json()) as { jobId?: string; status?: string; error?: string };
      if (!res.ok) throw new Error(data.error ?? "Failed to start job");
      setJobId(data.jobId ?? null);
      setJobStatus((data.status as JobStatus) ?? "queued");
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Something went wrong");
    }
  }

  // ── Active lyric line ─────────────────────────────────────────────────────────

  const activeLine = lyrics.find(
    (l) =>
      l.startTime !== null &&
      l.endTime !== null &&
      currentTime >= l.startTime &&
      currentTime <= l.endTime
  );

  // ── Render ────────────────────────────────────────────────────────────────────

  const isActive = jobStatus === "queued" || jobStatus === "processing";
  const hasFailed = jobStatus === "failed";
  const isDone = jobStatus === "completed";

  return (
    <div className="flex flex-col gap-6">
      {/* YouTube embed */}
      <div className="relative aspect-video w-full overflow-hidden rounded-xl bg-muted">
        <iframe
          src={`https://www.youtube.com/embed/${youtubeId}?enablejsapi=1`}
          title={title ?? "YouTube video"}
          className="absolute inset-0 h-full w-full"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        />
      </div>

      {/* Song info */}
      <div>
        <h1 className="text-2xl font-bold">{title ?? "Unknown Title"}</h1>
        {artist && <p className="text-muted-foreground">{artist}</p>}
      </div>

      {/* Processing panel */}
      {!isDone && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Zap className="h-4 w-4 text-primary" />
              {isActive ? "Processing…" : "Process Lyrics"}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {isActive && job ? (
              <>
                <div className="space-y-3">
                  {job.steps.map((step) => (
                    <StepRow key={step.name} step={step} />
                  ))}
                </div>
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Overall: {job.progress}%
                </div>
              </>
            ) : hasFailed ? (
              <div className="space-y-3">
                <p className="text-sm text-destructive">
                  Processing failed: {job?.error ?? "unknown error"}
                </p>
                <Button onClick={submitJob} variant="outline" size="sm">
                  Retry
                </Button>
              </div>
            ) : (
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Extract lyrics from this video using local Whisper + WhisperX forced alignment.
                  No paid APIs — runs entirely on your machine.
                </p>
                {submitError && (
                  <p className="text-sm text-destructive">{submitError}</p>
                )}
                <Button onClick={submitJob} className="gap-2">
                  <Zap className="h-4 w-4" />
                  Start Processing
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Lyrics display */}
      {lyrics.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center justify-between text-base">
              <span>Lyrics</span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsPlaying((p) => !p)}
                className="gap-1.5"
              >
                {isPlaying ? (
                  <Pause className="h-4 w-4" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
                {isPlaying ? "Pause" : "Play"}
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-1">
              {lyrics.map((line) => (
                <div
                  key={line.index}
                  className={`rounded-lg px-4 py-3 transition-all duration-300 ${
                    activeLine?.index === line.index
                      ? "animate-karaoke-highlight bg-primary/10 text-primary"
                      : "text-foreground/70 hover:bg-muted"
                  }`}
                >
                  <p
                    className="font-japanese text-lg leading-loose"
                    dangerouslySetInnerHTML={{ __html: line.japanese }}
                  />
                  {line.analysis?.translation.natural && (
                    <p className="mt-1 text-sm text-muted-foreground">
                      {line.analysis.translation.natural}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
