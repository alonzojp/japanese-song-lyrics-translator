"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Play, Pause, Loader2, CheckCircle2, XCircle, Zap,
  ChevronDown, ChevronUp, Terminal, BookOpen, MousePointer2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AnkiExportDialog } from "@/components/anki-export-dialog";
import { VocabCard, makeWordCard } from "@/components/vocab-card";
import type { Job, JobLogEntry, JobStepInfo, JobStatus, LyricLine, Token } from "@japanese-lyrics/shared";
import type { ExportCard } from "@japanese-lyrics/anki";

// ── YouTube IFrame API types ───────────────────────────────────────────────────

interface YTPlayer {
  playVideo(): void;
  pauseVideo(): void;
  getCurrentTime(): number;
  getPlayerState(): number;
  destroy(): void;
}

declare global {
  interface Window {
    YT: { Player: new (el: HTMLElement, opts: unknown) => YTPlayer; PlayerState: Record<string, number> };
    onYouTubeIframeAPIReady: () => void;
  }
}

// ── Types ──────────────────────────────────────────────────────────────────────

interface KaraokePlayerProps {
  songId:        string;
  youtubeId:     string;
  title:         string | null;
  artist:        string | null;
  initialJobId:  string | null;
  initialStatus: string | null;
  initialLyrics: LyricLine[];
}

interface VocabState {
  token:      Token;
  tokenIndex: number;
  line:       LyricLine;
}

interface SelectedItem {
  token:    Token;
  lineText: string;
  startTime: number;
  key:      string;   // `${line.id}-${tokenIndex}`
}

const STEP_LABELS: Record<string, string> = {
  download:   "Download audio",
  separate:   "Separate vocals",
  transcribe: "Transcribe speech",
  align:      "Align timestamps",
};

const POLL_MS    = 1200;
const LEAD_IN_S  = 1.2;    // show next line N seconds before its acoustic start
const LINGER_S   = 0.35;   // keep previous line visible for N seconds after transition

// ── Step row ───────────────────────────────────────────────────────────────────

function StepRow({ step }: { step: JobStepInfo }) {
  const icon =
    step.status === "completed"  ? <CheckCircle2 className="h-4 w-4 shrink-0 text-green-500" /> :
    step.status === "processing" ? <Loader2       className="h-4 w-4 shrink-0 animate-spin text-primary" /> :
    step.status === "failed"     ? <XCircle       className="h-4 w-4 shrink-0 text-destructive" /> :
                                   <div className="h-4 w-4 shrink-0 rounded-full border border-muted-foreground/30" />;

  return (
    <div className="flex items-start gap-3">
      <div className="mt-0.5">{icon}</div>
      <div className="flex-1 space-y-1">
        <div className="flex items-center justify-between text-sm">
          <span
            className={
              step.status === "processing" ? "font-medium text-foreground" :
              step.status === "completed"  ? "text-muted-foreground" :
              "text-muted-foreground/50"
            }
          >
            {STEP_LABELS[step.name] ?? step.label}
          </span>
          {step.status === "processing" && (
            <span className="text-xs text-muted-foreground">{step.progress}%</span>
          )}
          {step.status === "completed" && (
            <span className="text-xs text-green-600">done</span>
          )}
        </div>
        {step.status === "processing" && step.message && (
          <p className="text-xs text-muted-foreground">{step.message}</p>
        )}
        {step.status === "processing" && (
          <div className="h-1 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-primary transition-all duration-500"
              style={{ width: `${step.progress}%` }}
            />
          </div>
        )}
        {step.error && (
          <p className="text-xs text-destructive">{step.error}</p>
        )}
      </div>
    </div>
  );
}

// ── Log panel ──────────────────────────────────────────────────────────────────

function LogPanel({ logs }: { logs: JobLogEntry[] }) {
  const [open, setOpen]   = useState(false);
  const containerRef      = useRef<HTMLDivElement>(null);

  // Scroll the log container itself, not the page
  useEffect(() => {
    if (open && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs, open]);

  if (logs.length === 0) return null;

  return (
    <div className="rounded-lg border border-border bg-muted/30">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-4 py-2 text-xs text-muted-foreground hover:text-foreground"
      >
        <span className="flex items-center gap-1.5">
          <Terminal className="h-3.5 w-3.5" />
          Processing log ({logs.length} entries)
        </span>
        {open ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
      </button>

      {open && (
        <div
          ref={containerRef}
          className="max-h-52 overflow-y-auto border-t border-border px-4 py-2 font-mono"
        >
          {logs.map((entry, i) => (
            <div key={i} className="flex gap-2 py-0.5 text-xs">
              <span className="shrink-0 text-muted-foreground/60">
                {new Date(entry.ts).toLocaleTimeString()}
              </span>
              {entry.stage && (
                <span className="shrink-0 text-primary/70">[{entry.stage}]</span>
              )}
              <span
                className={
                  entry.level === "error"   ? "text-destructive" :
                  entry.level === "warning" ? "text-yellow-500"  :
                  "text-foreground/80"
                }
              >
                {entry.message}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Token span ─────────────────────────────────────────────────────────────────

function TokenSpan({
  token,
  isSelected,
  isHighlighted,
  onClick,
}: {
  token:        Token;
  isSelected:   boolean;
  isHighlighted: boolean;
  onClick:      () => void;
}) {
  const base =
    "cursor-pointer rounded px-[1px] transition-colors select-none " +
    "hover:bg-primary/15 active:bg-primary/25 " +
    (isSelected   ? "bg-primary/20 text-primary ring-1 ring-primary/40 " : "") +
    (isHighlighted ? "text-primary " : "");

  return (
    <span className={base} onClick={onClick}>
      {token.furigana ? (
        <ruby>
          {token.surface}
          <rt className="text-[0.45em] font-normal text-muted-foreground">
            {token.reading}
          </rt>
        </ruby>
      ) : (
        token.surface
      )}
    </span>
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
  // ── Job / lyrics state ────────────────────────────────────────────────────────
  const [jobId, setJobId]           = useState<string | null>(initialJobId);
  const [jobStatus, setJobStatus]   = useState<JobStatus | null>(initialStatus as JobStatus | null);
  const [job, setJob]               = useState<Job | null>(null);
  const [lyrics, setLyrics]         = useState<LyricLine[]>(initialLyrics);
  const [isPlaying, setIsPlaying]       = useState(false);
  const [currentTime, setCurrentTime]   = useState(0);
  const [submitError, setSubmitError]   = useState<string | null>(null);
  const [isSyncingLyrics, setIsSyncing]         = useState(false);
  const [isFetchingLyrics, setIsFetchingLyrics] = useState(false);
  const [fetchLyricsError, setFetchLyricsError] = useState<string | null>(null);
  const [isPastingLyrics, setIsPasting]         = useState(false);
  const [pasteText, setPasteText]       = useState("");
  const [pasteError, setPasteError]     = useState<string | null>(null);
  const [isPasteLoading, setIsPasteLoading] = useState(false);
  const pollRef                         = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── YouTube player refs ───────────────────────────────────────────────────────
  const ytPlayerRef       = useRef<YTPlayer | null>(null);
  const playerContainerRef = useRef<HTMLDivElement>(null);
  const timeIntervalRef   = useRef<ReturnType<typeof setInterval> | null>(null);
  const activeLineRef     = useRef<HTMLDivElement | null>(null);

  // ── Vocab / selection state ───────────────────────────────────────────────────
  const [vocabState, setVocabState]         = useState<VocabState | null>(null);
  const [selectionMode, setSelectionMode]   = useState(false);
  const [selectedItems, setSelectedItems]   = useState<SelectedItem[]>([]);
  const [exportCards, setExportCards]       = useState<ExportCard[] | null>(null);
  const [isAnalyzing, setIsAnalyzing]       = useState(false);
  const [analyzeError, setAnalyzeError]     = useState<string | null>(null);

  // ── Derived ───────────────────────────────────────────────────────────────────
  const hasTokens  = lyrics.some((l) => l.tokens && l.tokens.length > 0);

  // Clamp endTimes so no line extends past the next line's start.
  // This eliminates DTW-produced overlaps before any display logic sees them.
  const processedLyrics = useMemo(() =>
    lyrics.map((line, i) => {
      const nextStart  = lyrics[i + 1]?.startTime;
      const clampedEnd = nextStart !== undefined
        ? Math.min(line.endTime, nextStart - 0.05)
        : line.endTime;
      if (process.env.NODE_ENV !== "production") {
        const dur = clampedEnd - line.startTime;
        if (dur > 12) {
          console.warn(
            `LONG_LINE_WARNING: line ${i} "${line.text.slice(0, 30)}" ` +
            `visual_duration=${dur.toFixed(1)}s (raw_end=${line.endTime.toFixed(3)})`
          );
        }
      }
      return clampedEnd === line.endTime ? line : { ...line, endTime: clampedEnd };
    }),
    [lyrics],
  );

  // Active line: last line whose clamped window contains currentTime.
  // reduce() scans all lines so the highest-index match wins — handles any residual overlap.
  const activeIdx = processedLyrics.reduce<number>(
    (best, l, i) =>
      currentTime >= l.startTime && currentTime <= l.endTime ? i : best,
    -1,
  );
  const activeLine = activeIdx >= 0 ? processedLyrics[activeIdx] : null;

  // Incoming line: the next line, shown early to give the learner a reading head-start.
  const incomingLine: LyricLine | null =
    activeIdx >= 0 && activeIdx + 1 < processedLyrics.length &&
    currentTime >= processedLyrics[activeIdx + 1].startTime - LEAD_IN_S
      ? processedLyrics[activeIdx + 1]
      : null;

  // Lingering line: the previous line, briefly kept visible after a transition.
  const lingeringLine: LyricLine | null =
    activeIdx > 0 && activeLine &&
    currentTime < activeLine.startTime + LINGER_S
      ? processedLyrics[activeIdx - 1]
      : null;

  const isActive    = jobStatus === "queued" || jobStatus === "processing";
  const isDone      = jobStatus === "completed";
  const hasFailed   = jobStatus === "failed";
  const overallPct  = job?.progress ?? 0;
  const logs        = job?.recentLogs ?? [];

  // ── Job polling ───────────────────────────────────────────────────────────────

  const reloadLyrics = useCallback(async () => {
    setIsSyncing(true);
    try {
      const res = await fetch(`/api/songs/${songId}/lyrics`);
      if (res.ok) {
        const { lines } = (await res.json()) as { lines: LyricLine[] };
        setLyrics(lines);
      }
    } finally {
      setIsSyncing(false);
    }
  }, [songId]);

  const poll = useCallback(async (id: string) => {
    try {
      const res  = await fetch(`/api/jobs/${id}`);
      if (!res.ok) {
        pollRef.current = setTimeout(() => poll(id), POLL_MS);
        return;
      }
      const data = (await res.json()) as Job;
      setJob(data);
      setJobStatus(data.status);

      if (data.status === "completed") {
        await reloadLyrics();
      } else if (data.status === "processing" || data.status === "queued") {
        pollRef.current = setTimeout(() => poll(id), POLL_MS);
      }
    } catch {
      pollRef.current = setTimeout(() => poll(id), POLL_MS);
    }
  }, [reloadLyrics]);

  useEffect(() => {
    if (jobId && (jobStatus === "processing" || jobStatus === "queued")) poll(jobId);
    return () => { if (pollRef.current) clearTimeout(pollRef.current); };
  }, [jobId, jobStatus, poll]);

  // If job is already completed but lyrics weren't saved (polling missed the finish),
  // sync automatically on mount so the user never has to click anything.
  useEffect(() => {
    if (initialJobId && initialStatus === "completed" && initialLyrics.length === 0) {
      fetch(`/api/jobs/${initialJobId}`)
        .then((r) => r.ok ? reloadLyrics() : null)
        .catch(() => null);
    }
  }, [initialJobId, initialStatus, initialLyrics.length, reloadLyrics]);

  // ── YouTube IFrame API ────────────────────────────────────────────────────────

  useEffect(() => {
    if (typeof window === "undefined") return;

    const startInterval = () => {
      if (timeIntervalRef.current) clearInterval(timeIntervalRef.current);
      timeIntervalRef.current = setInterval(() => {
        if (ytPlayerRef.current) setCurrentTime(ytPlayerRef.current.getCurrentTime());
      }, 100);
    };

    const stopInterval = () => {
      if (timeIntervalRef.current) { clearInterval(timeIntervalRef.current); timeIntervalRef.current = null; }
    };

    const initPlayer = () => {
      if (!playerContainerRef.current || !window.YT?.Player) return;
      ytPlayerRef.current = new window.YT.Player(playerContainerRef.current, {
        videoId: youtubeId,
        playerVars: { enablejsapi: 1, origin: window.location.origin },
        events: {
          onStateChange: (e: { data: number }) => {
            if (e.data === 1) { setIsPlaying(true);  startInterval(); }
            else              { setIsPlaying(false); stopInterval();  }
          },
        },
      });
    };

    if (window.YT?.Player) {
      initPlayer();
    } else {
      window.onYouTubeIframeAPIReady = initPlayer;
      if (!document.querySelector('script[src*="youtube.com/iframe_api"]')) {
        const tag = document.createElement("script");
        tag.src = "https://www.youtube.com/iframe_api";
        document.head.appendChild(tag);
      }
    }

    return () => {
      stopInterval();
      ytPlayerRef.current?.destroy();
      ytPlayerRef.current = null;
    };
  }, [youtubeId]);

  // ── Auto-scroll active lyric into view ────────────────────────────────────────

  useEffect(() => {
    activeLineRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [activeLine?.id]);

  // ── Submit processing job ─────────────────────────────────────────────────────

  async function submitJob() {
    setSubmitError(null);
    try {
      const res  = await fetch("/api/jobs", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ songId }),
      });
      const data = (await res.json()) as { jobId?: string; status?: string; error?: string };
      if (!res.ok) throw new Error(data.error ?? "Failed to start job");
      setJobId(data.jobId ?? null);
      setJobStatus((data.status as JobStatus) ?? "queued");
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Something went wrong");
    }
  }

  // ── NLP analysis ─────────────────────────────────────────────────────────────

  async function handleAnalyze() {
    setIsAnalyzing(true);
    setAnalyzeError(null);
    try {
      const res = await fetch(`/api/songs/${songId}/analyze`, { method: "POST" });
      if (!res.ok) {
        const { error } = (await res.json().catch(() => ({}))) as { error?: string };
        throw new Error(error ?? "Analysis failed");
      }
      await reloadLyrics();
    } catch (err) {
      setAnalyzeError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setIsAnalyzing(false);
    }
  }

  // ── Re-fetch lyrics from provider chain ──────────────────────────────────────

  async function handleFetchLyrics() {
    setIsFetchingLyrics(true);
    setFetchLyricsError(null);
    try {
      const res = await fetch("/api/lyrics", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          youtubeId,
          youtubeUrl: `https://www.youtube.com/watch?v=${youtubeId}`,
          songId,
          force: true,
        }),
      });
      const data = (await res.json()) as { lines?: unknown[]; error?: string };
      if (!res.ok) throw new Error(data.error ?? "Failed to fetch lyrics");
      if ((data.lines?.length ?? 0) > 0) {
        await reloadLyrics();
      } else {
        setFetchLyricsError("No lyrics found from any provider");
      }
    } catch (err) {
      setFetchLyricsError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setIsFetchingLyrics(false);
    }
  }

  // ── Paste official lyrics ─────────────────────────────────────────────────────

  async function handlePasteLyrics() {
    if (!pasteText.trim()) return;
    setIsPasteLoading(true);
    setPasteError(null);
    try {
      const res = await fetch(`/api/songs/${songId}/lyrics`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ text: pasteText }),
      });
      const data = (await res.json()) as { lines?: LyricLine[]; error?: string };
      if (!res.ok) throw new Error(data.error ?? "Failed to sync lyrics");
      setLyrics(data.lines ?? []);
      setIsPasting(false);
      setPasteText("");
    } catch (err) {
      setPasteError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setIsPasteLoading(false);
    }
  }

  // ── Token interaction ─────────────────────────────────────────────────────────

  function handleTokenClick(token: Token, tokenIndex: number, line: LyricLine) {
    if (selectionMode) {
      const key = `${line.id}-${tokenIndex}`;
      setSelectedItems((prev) => {
        const exists = prev.findIndex((s) => s.key === key);
        if (exists !== -1) return prev.filter((_, i) => i !== exists);
        return [
          ...prev,
          { token, lineText: line.text, startTime: line.startTime, key },
        ];
      });
    } else {
      setVocabState({ token, tokenIndex, line });
    }
  }

  function handleVocabExport(cards: ExportCard[]) {
    setVocabState(null);
    setExportCards(cards);
  }

  function clearSelection() {
    setSelectionMode(false);
    setSelectedItems([]);
  }

  function exportSelected() {
    const cards = selectedItems.map(({ token, lineText, startTime }) =>
      makeWordCard(token, lineText, title, startTime),
    );
    setExportCards(cards);
  }

  // ── Render ────────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col gap-6">

      {/* YouTube embed — div replaced by YT.Player iframe */}
      <div className="relative aspect-video w-full overflow-hidden rounded-xl bg-muted">
        <div ref={playerContainerRef} className="absolute inset-0 h-full w-full" />
      </div>

      {/* Song info */}
      <div>
        <h1 className="text-2xl font-bold">{title ?? "Unknown Title"}</h1>
        {artist && <p className="text-muted-foreground">{artist}</p>}
      </div>

      {/* Processing / status panel */}
      {!isDone && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Zap className="h-4 w-4 text-primary" />
              {isActive ? (
                <span className="flex items-center gap-2">
                  Processing…
                  <span className="text-sm font-normal text-muted-foreground">{overallPct}%</span>
                </span>
              ) : "Process Lyrics"}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {isActive && (
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-primary transition-all duration-700"
                  style={{ width: `${overallPct}%` }}
                />
              </div>
            )}

            {isActive && job ? (
              <>
                <div className="space-y-3">
                  {job.steps.map((step) => <StepRow key={step.name} step={step} />)}
                </div>
                <LogPanel logs={logs} />
              </>
            ) : hasFailed ? (
              <div className="space-y-3">
                <p className="text-sm text-destructive">
                  {job?.error ?? "Processing failed — unknown error"}
                </p>
                <LogPanel logs={logs} />
                <Button onClick={submitJob} variant="outline" size="sm">Retry</Button>
              </div>
            ) : (
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  Extract lyrics using Whisper AI. Audio is cached — re-submitting the same video is instant.
                </p>
                {submitError && <p className="text-sm text-destructive">{submitError}</p>}
                <Button onClick={submitJob} className="gap-2">
                  <Zap className="h-4 w-4" />
                  Start Processing
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Re-fetch lyrics (job done but no lyrics stored) */}
      {lyrics.length === 0 && !isActive && (
        <Card>
          <CardContent className="pt-6 space-y-3">
            <p className="text-sm text-muted-foreground">
              No lyrics found. Try fetching from Japanese lyrics databases (UtaNet, Utaten, PetitLyrics).
            </p>
            {fetchLyricsError && <p className="text-sm text-destructive">{fetchLyricsError}</p>}
            <Button onClick={handleFetchLyrics} disabled={isFetchingLyrics} className="gap-2">
              {isFetchingLyrics
                ? <><Loader2 className="h-4 w-4 animate-spin" />Fetching…</>
                : <><Zap className="h-4 w-4" />Fetch lyrics</>
              }
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Syncing indicator */}
      {isSyncingLyrics && (
        <div className="flex items-center gap-2 rounded-lg border px-4 py-3 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin text-primary" />
          Loading lyrics…
        </div>
      )}

      {/* Lyrics display */}
      {lyrics.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center justify-between text-base">
              <span>Lyrics</span>

              <div className="flex items-center gap-2">
                {/* Paste official lyrics */}
                {isDone && (
                  <Button
                    variant="ghost" size="sm"
                    onClick={() => setIsPasting((p) => !p)}
                    className="gap-1.5 text-muted-foreground"
                  >
                    <BookOpen className="h-3.5 w-3.5" />
                    {isPastingLyrics ? "Cancel" : "Paste lyrics"}
                  </Button>
                )}

                {/* Reprocess */}
                {isDone && (
                  <Button variant="ghost" size="sm" onClick={submitJob} className="gap-1.5 text-muted-foreground">
                    <Zap className="h-3.5 w-3.5" />
                    Reprocess
                  </Button>
                )}

                {/* Play/pause toggle */}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    if (isPlaying) ytPlayerRef.current?.pauseVideo();
                    else           ytPlayerRef.current?.playVideo();
                  }}
                  className="gap-1.5"
                >
                  {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                  {isPlaying ? "Pause" : "Play"}
                </Button>

                {/* Analyze vocabulary */}
                {!hasTokens && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    className="gap-1.5"
                  >
                    {isAnalyzing
                      ? <Loader2 className="h-4 w-4 animate-spin" />
                      : <BookOpen className="h-4 w-4" />
                    }
                    {isAnalyzing ? "Analyzing…" : "Analyze vocabulary"}
                  </Button>
                )}

                {/* Selection mode toggle */}
                {hasTokens && (
                  <Button
                    variant={selectionMode ? "default" : "outline"}
                    size="sm"
                    onClick={() => {
                      setSelectionMode((v) => !v);
                      setSelectedItems([]);
                    }}
                    className="gap-1.5"
                  >
                    <MousePointer2 className="h-4 w-4" />
                    {selectionMode ? "Done" : "Select"}
                  </Button>
                )}
              </div>
            </CardTitle>

            {analyzeError && (
              <p className="mt-1 text-xs text-destructive">{analyzeError}</p>
            )}

            {selectionMode && (
              <p className="mt-1 text-xs text-muted-foreground">
                Click words to add them to your export queue.
              </p>
            )}
          </CardHeader>

          {isPastingLyrics && (
            <div className="border-t px-6 py-4 space-y-3">
              <p className="text-xs text-muted-foreground">
                Paste the official lyrics below. The app will use AI alignment to sync them to the audio timestamps.
              </p>
              <textarea
                value={pasteText}
                onChange={(e) => setPasteText(e.target.value)}
                placeholder={"Paste Japanese lyrics here…\n一行目\n二行目\n…"}
                className="w-full rounded-md border bg-background px-3 py-2 text-sm font-japanese leading-relaxed outline-none focus:ring-2 focus:ring-primary/40 min-h-[160px] resize-y"
                disabled={isPasteLoading}
              />
              {pasteError && <p className="text-xs text-destructive">{pasteError}</p>}
              <Button
                onClick={handlePasteLyrics}
                disabled={isPasteLoading || !pasteText.trim()}
                size="sm"
                className="gap-2"
              >
                {isPasteLoading
                  ? <><Loader2 className="h-3.5 w-3.5 animate-spin" />Syncing timestamps…</>
                  : <><Zap className="h-3.5 w-3.5" />Sync lyrics</>
                }
              </Button>
            </div>
          )}

          <CardContent>
            <div className="space-y-1">
              {processedLyrics.map((line) => {
                const isCurrentLine  = activeLine?.id   === line.id;
                const isNextLine     = incomingLine?.id === line.id;
                const isPreviousLine = lingeringLine?.id === line.id;

                return (
                  <div
                    key={line.id}
                    ref={isCurrentLine ? activeLineRef : null}
                    className={`rounded-lg px-4 py-3 transition-all duration-300 ${
                      isCurrentLine
                        ? "bg-primary/20 ring-1 ring-primary/30"
                        : isNextLine
                        ? "bg-primary/10 ring-1 ring-primary/15 opacity-75"
                        : isPreviousLine
                        ? "bg-primary/8 opacity-40"
                        : "hover:bg-muted"
                    }`}
                  >
                    {line.tokens && line.tokens.length > 0 ? (
                      <p className="font-japanese text-lg leading-[2.4]">
                        {line.tokens.map((token, ti) => {
                          const selKey     = `${line.id}-${ti}`;
                          const isSelected = selectedItems.some((s) => s.key === selKey);
                          return (
                            <TokenSpan
                              key={ti}
                              token={token}
                              isSelected={isSelected}
                              isHighlighted={isCurrentLine}
                              onClick={() => handleTokenClick(token, ti, line)}
                            />
                          );
                        })}
                      </p>
                    ) : (
                      <p className={`font-japanese text-lg leading-loose ${
                        isCurrentLine ? "text-foreground" :
                        isNextLine    ? "text-foreground/70" :
                        "text-foreground/70"
                      }`}>
                        {line.text}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Selection tray (fixed bottom bar when items selected) */}
      {selectionMode && selectedItems.length > 0 && (
        <div className="fixed bottom-6 left-1/2 z-30 flex -translate-x-1/2 items-center gap-3 rounded-full bg-primary px-5 py-3 text-sm font-medium text-white shadow-xl">
          <span>
            {selectedItems.length} word{selectedItems.length !== 1 ? "s" : ""} selected
          </span>
          <button
            onClick={exportSelected}
            className="rounded-full bg-white/20 px-3 py-1 transition hover:bg-white/30 active:scale-95"
          >
            Export to Anki
          </button>
          <button
            onClick={clearSelection}
            className="rounded-full px-2 py-1 text-white/70 transition hover:text-white"
            aria-label="Clear selection"
          >
            ✕
          </button>
        </div>
      )}

      {/* Vocab card popup */}
      {vocabState && (
        <VocabCard
          token={vocabState.token}
          tokenIndex={vocabState.tokenIndex}
          lineTokens={vocabState.line.tokens!}
          lineText={vocabState.line.text}
          startTime={vocabState.line.startTime}
          songTitle={title}
          onClose={() => setVocabState(null)}
          onExport={handleVocabExport}
        />
      )}

      {/* Anki export dialog (shared by vocab card + batch export) */}
      {exportCards && (
        <AnkiExportDialog
          cards={exportCards}
          deckName={title ? `${title} – Japanese Lyrics` : "Japanese Lyrics"}
          open={true}
          onClose={() => {
            setExportCards(null);
            clearSelection();
          }}
        />
      )}
    </div>
  );
}
