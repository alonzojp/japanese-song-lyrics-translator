"use client";

// ── Build identity ─────────────────────────────────────────────────────────────
// Bump this string whenever playback logic changes so you can verify the
// deployed version in DevTools → Console without reading minified bundles.
const PLAYBACK_BUILD = "2026-05-24 antistall-v2";

// ── Anti-stall constants ───────────────────────────────────────────────────────
// Safety net for lines whose backend display window is longer than any real lyric.
// Backend already caps at 10 s; these fire only if a line somehow gets stuck.
const MAX_ACTIVE_DURATION_S = 10.0;   // force-exit after this long on one line
const LONG_PAUSE_ESCAPE_S   = 1.2;    // silence gap above which tighter cap applies
const LONG_PAUSE_MAX_MULT   = 0.6;    // tighter cap = MAX_ACTIVE_DURATION_S × this

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Play, Pause, Loader2, CheckCircle2, XCircle, Zap,
  ChevronDown, ChevronUp, Terminal, BookOpen, MousePointer2, Copy, Check,
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

const POLL_MS = 1200;

/**
 * Pure deterministic active-line lookup.
 *
 * Returns the index of the line whose [startTime, endTime] window contains t.
 * Returns -1 if t is in a gap, before all lines, or after all lines.
 *
 * Algorithm: binary search to the floor candidate (last line with startTime ≤ t),
 * then a single endTime check. O(log n). No state. Idempotent.
 */
function findActiveLine(lines: LyricLine[], t: number): number {
  if (!lines.length) return -1;

  // Binary search: find last index where startTime <= t
  let lo = 0, hi = lines.length - 1;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if ((lines[mid]?.startTime ?? Infinity) <= t) lo = mid; else hi = mid - 1;
  }

  // lo is now the candidate: last line whose start has been reached
  const candidate = lines[lo];
  if (!candidate || candidate.startTime > t) return -1;  // t is before all lines
  if (t <= candidate.endTime)                return lo;   // t is inside this line
  return -1;                                              // t is in a gap after this line
}

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
  const [open, setOpen]        = useState(false);
  const [copied, setCopied]    = useState(false);
  const containerRef           = useRef<HTMLDivElement>(null);
  const userScrolledUpRef      = useRef(false);

  function copyLogs() {
    try {
      const text = logs
        .map((e) => `${new Date(e.ts).toLocaleTimeString()} [${e.stage ?? "—"}] ${e.message}`)
        .join("\n");
      const finish = () => { setCopied(true); setTimeout(() => setCopied(false), 2000); };
      if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
        navigator.clipboard.writeText(text).then(finish).catch(() => {
          try { fallbackCopy(text, finish); } catch (_e) { /* silent */ }
        });
      } else {
        try { fallbackCopy(text, finish); } catch (_e) { /* silent */ }
      }
    } catch (_e) {
      /* silent — never crash the app over a copy failure */
    }
  }

  function fallbackCopy(text: string, onDone: () => void) {
    const el = document.createElement("textarea");
    el.value = text;
    el.style.cssText = "position:fixed;top:-9999px;opacity:0;pointer-events:none";
    document.body.appendChild(el);
    el.focus();
    el.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(el);
    if (ok) onDone();
  }

  // Reset intent flag when panel opens so it starts pinned to bottom
  useEffect(() => {
    if (open) userScrolledUpRef.current = false;
  }, [open]);

  // Detect when the user manually scrolls away from the bottom
  const handleScroll = () => {
    const c = containerRef.current;
    if (!c) return;
    userScrolledUpRef.current = c.scrollHeight - c.scrollTop - c.clientHeight > 60;
  };

  // Auto-scroll only when user has not scrolled up
  useEffect(() => {
    if (!open || !containerRef.current || userScrolledUpRef.current) return;
    containerRef.current.scrollTop = containerRef.current.scrollHeight;
  }, [logs, open]);

  if (logs.length === 0) return null;

  return (
    <div className="rounded-lg border border-border bg-muted/30">
      <div className="flex items-center justify-between px-4 py-2">
        <button
          onClick={() => setOpen((o) => !o)}
          className="flex flex-1 items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground"
        >
          <Terminal className="h-3.5 w-3.5" />
          Processing log ({logs.length} entries)
          {process.env.NODE_ENV !== "production" && (
            <span className="ml-1 text-primary/50">[rendered {logs.length}]</span>
          )}
          {open ? <ChevronUp className="h-3.5 w-3.5 ml-1" /> : <ChevronDown className="h-3.5 w-3.5 ml-1" />}
        </button>
        <button
          onClick={copyLogs}
          className="flex items-center gap-1 rounded px-2 py-0.5 text-xs text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
          title="Copy all logs"
        >
          {copied ? <Check className="h-3 w-3 text-green-500" /> : <Copy className="h-3 w-3" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>

      {open && (
        <div
          ref={containerRef}
          onScroll={handleScroll}
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
  const [submitError, setSubmitError]   = useState<string | null>(null);
  const [isSyncingLyrics, setIsSyncing]         = useState(false);
  const [isFetchingLyrics, setIsFetchingLyrics] = useState(false);
  const [fetchLyricsError, setFetchLyricsError] = useState<string | null>(null);
  const [isPastingLyrics, setIsPasting]         = useState(false);
  const [pasteText, setPasteText]       = useState("");
  const [pasteError, setPasteError]     = useState<string | null>(null);
  const [isPasteLoading, setIsPasteLoading] = useState(false);
  const [completedLogs, setCompletedLogs] = useState<JobLogEntry[]>([]);
  const [showTiming, setShowTiming]       = useState(false);
  const pollRef                         = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── YouTube player refs ───────────────────────────────────────────────────────
  const ytPlayerRef        = useRef<YTPlayer | null>(null);
  const playerContainerRef = useRef<HTMLDivElement>(null);
  const activeLineRef      = useRef<HTMLDivElement | null>(null);

  // ── rAF-based timing refs — never stored in React state ──────────────────────
  // The rAF loop reads all of these directly, so it never captures a stale
  // closure and never needs to be recreated when component state changes.
  const rafHandleRef      = useRef<number>(0);
  const currentTimeRef    = useRef<number>(0);
  const activeIdxRef      = useRef<number>(-1);
  const isPlayingRef      = useRef<boolean>(false);
  const antiStallLineRef  = useRef<number>(-1);   // tracks which line triggered anti-stall (suppress repeat logs)
  // Initialized with initialLyrics so the loop is correct on the very first tick,
  // before any lyrics-change effect has fired.
  const lyricsRef      = useRef<LyricLine[]>(initialLyrics);

  // ── Vocab / selection state ───────────────────────────────────────────────────
  const [vocabState, setVocabState]         = useState<VocabState | null>(null);
  const [selectionMode, setSelectionMode]   = useState(false);
  const [selectedItems, setSelectedItems]   = useState<SelectedItem[]>([]);
  const [exportCards, setExportCards]       = useState<ExportCard[] | null>(null);
  const [isAnalyzing, setIsAnalyzing]       = useState(false);
  const [analyzeError, setAnalyzeError]     = useState<string | null>(null);

  // ── Derived ───────────────────────────────────────────────────────────────────
  const hasTokens  = lyrics.some((l) => l.tokens && l.tokens.length > 0);

  // lyrics is already sorted and finalized by the backend. No client-side mutations.
  const processedLyrics = lyrics;

  // Mirror lyrics into a ref so the rAF loop reads the latest array without
  // capturing a stale closure.  Also resets the active index so a highlight
  // from a previous lyric set can never briefly appear over fresh lyrics.
  useEffect(() => {
    lyricsRef.current    = lyrics;
    activeIdxRef.current = -1;
    setActiveIdx(-1);
  }, [lyrics]);

  // activeIdx is driven by the rAF loop, not by useMemo.
  // React state is updated only when the active line actually changes,
  // so renders happen at line-change frequency (~0.5–3 Hz) rather than
  // at clock-poll frequency (100 Hz with setInterval, 60 Hz with rAF).
  const [activeIdx, setActiveIdx] = useState(-1);

  const activeLine = activeIdx >= 0 ? processedLyrics[activeIdx] : null;

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
        // Fetch best logs via song endpoint (bypasses currentJobId race)
        fetch(`/api/songs/${songId}/logs`)
          .then((r) => r.ok ? r.json() : null)
          .then((d: { logs?: JobLogEntry[]; logCount?: number } | null) => {
            console.log(`[log_debug] API returned ${d?.logCount ?? 0} log entries`);
            if (d?.logs?.length) setCompletedLogs(d.logs);
          })
          .catch(() => null);
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

  // Fetch full logs on mount — always uses song-level endpoint which finds
  // the most log-rich job, not just currentJobId (avoids instant-cache shadowing).
  useEffect(() => {
    fetch(`/api/songs/${songId}/logs`)
      .then((r) => r.ok ? r.json() : null)
      .then((data: { logs?: JobLogEntry[]; logCount?: number } | null) => {
        console.log(`[log_debug] mount: API returned ${data?.logCount ?? 0} entries`);
        if (data?.logs?.length) setCompletedLogs(data.logs);
      })
      .catch(() => null);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);  // intentionally runs once on mount only

  // ── YouTube IFrame API ────────────────────────────────────────────────────────

  useEffect(() => {
    if (typeof window === "undefined") return;

    // ── rAF clock loop ───────────────────────────────────────────────────────
    // Declared as a named function expression inside the effect so it can
    // reference itself by name for self-scheduling — no useCallback needed,
    // no stale-closure risk because every value is read through a ref.
    //
    // The loop runs continuously from mount to unmount.  isPlayingRef gates
    // whether getCurrentTime() is called; when paused, the loop idles at
    // ~0 μs/frame (one ref read + one rAF schedule).  This is cheaper than
    // starting and stopping the loop on every play/pause event.
    function tick() {
      rafHandleRef.current = requestAnimationFrame(tick);

      if (!isPlayingRef.current || !ytPlayerRef.current) return;

      const t = ytPlayerRef.current.getCurrentTime();
      currentTimeRef.current = t;

      // findActiveLine is O(log n) and stateless — safe to call every frame.
      const idx = findActiveLine(lyricsRef.current, t);

      // Anti-stall safety net: if a line has been active longer than
      // MAX_ACTIVE_DURATION_S, force-exit it regardless of endTime.
      // The backend already caps display windows at 10 s; this only fires
      // when something slips through (e.g. very long contiguous lines).
      // When the preceding silence gap exceeds LONG_PAUSE_ESCAPE_S the cap
      // tightens to 60 % — lines after pauses have no excuse to linger.
      let displayIdx = idx;
      if (idx !== -1) {
        const line     = lyricsRef.current[idx];
        const prevLine = idx > 0 ? lyricsRef.current[idx - 1] : null;
        const silenceGap   = prevLine ? Math.max(0, (line.acousticStart ?? line.startTime) - prevLine.endTime) : 0;
        const backendWindow = line.endTime - line.startTime;
        const maxDuration  = silenceGap > LONG_PAUSE_ESCAPE_S
          ? Math.max(MAX_ACTIVE_DURATION_S * LONG_PAUSE_MAX_MULT, backendWindow)
          : Math.max(MAX_ACTIVE_DURATION_S, backendWindow);
        const activeDuration = t - line.startTime;
        if (activeDuration > maxDuration) {
          displayIdx = -1;
          if (antiStallLineRef.current !== idx) {
            antiStallLineRef.current = idx;
            console.warn(
              `[karaoke-antistall] L${idx} forced exit: ` +
              `active=${activeDuration.toFixed(2)}s max=${maxDuration.toFixed(1)}s ` +
              `silenceGap=${silenceGap.toFixed(2)}s endTime=${line.endTime.toFixed(3)}s`
            );
          }
        } else {
          antiStallLineRef.current = -1;
        }
      }

      // Only schedule a React re-render when the active line actually changes.
      // This is the key difference from the setInterval approach: the clock
      // ticks at 60 Hz but React renders at line-change frequency only.
      if (displayIdx !== activeIdxRef.current) {
        activeIdxRef.current = displayIdx;
        setActiveIdx(displayIdx);
      }
    }
    rafHandleRef.current = requestAnimationFrame(tick);

    // ── YouTube IFrame player ────────────────────────────────────────────────
    const initPlayer = () => {
      if (!playerContainerRef.current || !window.YT?.Player) return;
      ytPlayerRef.current = new window.YT.Player(playerContainerRef.current, {
        videoId: youtubeId,
        playerVars: { enablejsapi: 1, origin: window.location.origin },
        events: {
          onStateChange: (e: { data: number }) => {
            // 1 = YT.PlayerState.PLAYING; all other states (paused, buffering,
            // ended) stop the clock.  Both the ref and the state are updated
            // together: the ref gates the rAF loop; the state drives the UI.
            const playing = e.data === 1;
            isPlayingRef.current = playing;
            setIsPlaying(playing);
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
      cancelAnimationFrame(rafHandleRef.current);
      ytPlayerRef.current?.destroy();
      ytPlayerRef.current = null;
    };
  }, [youtubeId]);

  // ── Build identity (check in DevTools → Console) ─────────────────────────────
  useEffect(() => {
    console.log(`[karaoke] playback build: ${PLAYBACK_BUILD}`);
  }, []);

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

  // Reprocess: bust the alignment cache first so force_align + text-first
  // reconstruction actually re-run, then submit a new job.
  async function reprocess() {
    setSubmitError(null);
    try {
      const res = await fetch(`/api/cache/${youtubeId}/force-align`, { method: "DELETE" });
      if (!res.ok) {
        const { error } = (await res.json().catch(() => ({}))) as { error?: string };
        // 404 = not yet cached; that's fine, just submit the job directly
        if (res.status !== 404) throw new Error(error ?? "Cache invalidation failed");
      }
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Failed to clear cache");
      return;
    }
    await submitJob();
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

      {/* Log panel — shown during processing and persisted after completion */}
      {isActive && job && <LogPanel logs={logs} />}
      {isDone && completedLogs.length > 0 && <LogPanel logs={completedLogs} />}

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
              <div className="space-y-3">
                {job.steps.map((step) => <StepRow key={step.name} step={step} />)}
              </div>
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

      {/* Timing debug panel */}
      {lyrics.length > 0 && (
        <div>
          <button
            onClick={() => setShowTiming((v) => !v)}
            className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            {showTiming ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
            {showTiming ? "Hide timing" : "Show timing"}
          </button>
          {showTiming && (
            <div className="mt-2 overflow-x-auto rounded-lg border text-xs font-mono">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="border-b bg-muted/50 text-left">
                    <th className="px-3 py-2 text-muted-foreground">#</th>
                    <th className="px-3 py-2 text-muted-foreground">Start</th>
                    <th className="px-3 py-2 text-muted-foreground">End</th>
                    <th className="px-3 py-2 text-muted-foreground">Dur</th>
                    <th className="px-3 py-2 text-muted-foreground">Gap</th>
                    <th className="px-3 py-2 text-muted-foreground">Text</th>
                  </tr>
                </thead>
                <tbody>
                  {lyrics.map((line, i) => {
                    const st       = Number(line.startTime);
                    const et       = Number(line.endTime);
                    const prev     = i > 0 ? lyrics[i - 1] : null;
                    const gap      = prev != null ? st - Number(prev.endTime) : null;
                    const dur      = et - st;
                    const isActive = activeIdx === i;
                    return (
                      <tr
                        key={line.id ?? i}
                        className={`border-b last:border-0 ${isActive ? "bg-primary/10" : "hover:bg-muted/40"}`}
                      >
                        <td className="px-3 py-1.5 text-muted-foreground">{i.toString().padStart(2, "0")}</td>
                        <td className="px-3 py-1.5 tabular-nums">{st.toFixed(3)}</td>
                        <td className="px-3 py-1.5 tabular-nums">{et.toFixed(3)}</td>
                        <td className={`px-3 py-1.5 tabular-nums ${dur >= 9.9 ? "text-amber-500 font-semibold" : ""}`}>
                          {dur.toFixed(2)}s
                        </td>
                        <td className="px-3 py-1.5 tabular-nums text-muted-foreground">
                          {gap != null ? (gap < 0.01 ? "—" : `+${gap.toFixed(2)}s`) : ""}
                        </td>
                        <td className="px-3 py-1.5 font-japanese max-w-xs truncate">{line.text}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
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
                  <Button variant="ghost" size="sm" onClick={reprocess} className="gap-1.5 text-muted-foreground">
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
                const isCurrentLine = activeLine?.id === line.id;

                return (
                  <div
                    key={line.id}
                    ref={isCurrentLine ? activeLineRef : null}
                    className={`rounded-lg px-4 py-3 transition-all duration-300 ${
                      isCurrentLine
                        ? "bg-primary/20 ring-1 ring-primary/30"
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
                        isCurrentLine ? "text-foreground" : "text-foreground/70"
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
