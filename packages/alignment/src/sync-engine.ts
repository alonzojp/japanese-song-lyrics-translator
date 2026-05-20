/**
 * SyncEngine — real-time karaoke synchronization with anti-drift correction.
 *
 * The engine sits between the YouTube player (source of truth for time)
 * and the lyric display (consumer of active line index).
 *
 * Drift sources:
 *   - YouTube IFrame API timestamp granularity (~250ms polling)
 *   - WhisperX timing that's slightly off from actual vocals
 *   - Video preload / seek latency
 *
 * Anti-drift strategy:
 *   1. Exponential smoothing of the running offset (fast correction).
 *   2. Hard re-anchor when drift exceeds MAX_HARD_DRIFT (instant correction).
 *   3. Periodic re-anchor every REANCHOR_INTERVAL_MS ms using the nearest
 *      high-confidence line as a reference.
 *
 * Usage:
 *   const engine = new SyncEngine(alignedLines);
 *   // in the YouTube player tick (every ~250ms):
 *   const active = engine.getActiveLine(player.getCurrentTime());
 *   // when user manually clicks a line:
 *   engine.reportAnchor(lineId, player.getCurrentTime());
 */
import type { AlignedLine } from "./types.js";

const SMOOTH_ALPHA          = 0.15;   // EMA weight for offset correction (lower = smoother)
const MAX_HARD_DRIFT        = 2.5;    // seconds — hard re-anchor threshold
const REANCHOR_INTERVAL_MS  = 8000;   // re-anchor attempt every 8 s
const MIN_CONFIDENCE_ANCHOR = 0.55;   // minimum line confidence for auto re-anchor

export interface DriftEvent {
  ts:              number;    // wall-clock ms
  playbackTime:    number;    // player-reported time
  expectedTime:    number;    // what we expected
  rawDrift:        number;    // playbackTime - expectedTime
  appliedOffset:   number;    // offset after this correction
  type:            "smooth" | "hard_reanchor" | "manual_anchor" | "seek";
}

export interface SyncState {
  activeLineId:    string | null;
  activeIndex:     number;
  currentOffset:   number;
  lastDriftEvent:  DriftEvent | null;
  isReliable:      boolean;   // false if drift has been large recently
}

export class SyncEngine {
  private lines:       AlignedLine[];
  private offset:      number  = 0;      // seconds to subtract from playback time
  private lastAnchorMs: number  = 0;
  private driftHistory: number[] = [];
  private readonly maxHistLen  = 20;

  readonly driftLog: DriftEvent[] = [];

  constructor(lines: AlignedLine[]) {
    // Sort defensively
    this.lines = [...lines].sort((a, b) => a.startTime - b.startTime);
  }

  // ── Primary API ───────────────────────────────────────────────────────────────

  /** Return the line that should be highlighted at `playbackTime`. */
  getActiveLine(playbackTime: number): AlignedLine | null {
    const t = playbackTime - this.offset;
    // Binary search for the last line with startTime <= t
    let lo = 0;
    let hi = this.lines.length - 1;
    let best = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (this.lines[mid]!.startTime <= t) { best = mid; lo = mid + 1; }
      else hi = mid - 1;
    }
    if (best < 0) return null;
    const line = this.lines[best]!;
    // Only return if we're still within the line's endTime
    return line.endTime >= t ? line : null;
  }

  /** 0-based index of the active line (-1 if none). */
  getActiveIndex(playbackTime: number): number {
    const active = this.getActiveLine(playbackTime);
    if (!active) return -1;
    return this.lines.findIndex((l) => l.id === active.id);
  }

  /** Full sync state snapshot. */
  getState(playbackTime: number): SyncState {
    const active = this.getActiveLine(playbackTime);
    const recentDrift = this.driftHistory.slice(-5);
    const maxRecent   = recentDrift.length ? Math.max(...recentDrift.map(Math.abs)) : 0;
    return {
      activeLineId:   active?.id ?? null,
      activeIndex:    active ? this.lines.indexOf(active) : -1,
      currentOffset:  this.offset,
      lastDriftEvent: this.driftLog.at(-1) ?? null,
      isReliable:     maxRecent < 1.5,
    };
  }

  // ── Drift correction ──────────────────────────────────────────────────────────

  /**
   * Called by the player on each time update.
   * Optionally attempts auto re-anchoring using a nearby high-confidence line.
   */
  tick(playbackTime: number, nowMs: number = Date.now()): void {
    const active = this.getActiveLine(playbackTime);
    if (!active) return;

    // Auto re-anchor attempt
    if (nowMs - this.lastAnchorMs > REANCHOR_INTERVAL_MS) {
      this._autoReanchor(playbackTime, active, nowMs);
    }
  }

  /**
   * Report that at `actualPlaybackTime`, the given line should be active.
   * Used when the user manually selects a line, confirming its timestamp.
   */
  reportAnchor(lineId: string, actualPlaybackTime: number): void {
    const line = this.lines.find((l) => l.id === lineId);
    if (!line) return;

    const expected  = line.startTime + this.offset;
    const rawDrift  = actualPlaybackTime - expected;

    this._applyCorrection(rawDrift, actualPlaybackTime, expected, "manual_anchor");
    this.lastAnchorMs = Date.now();
  }

  /**
   * Call when the player seeks to a new position.
   * Resets drift tracking (seek invalidates accumulated offset).
   */
  onSeek(playbackTime: number): void {
    const event: DriftEvent = {
      ts:            Date.now(),
      playbackTime,
      expectedTime:  playbackTime,
      rawDrift:      0,
      appliedOffset: this.offset,
      type:          "seek",
    };
    this.driftLog.push(event);
    this.driftHistory = [];    // clear history on seek
  }

  /**
   * Replace the line list (e.g. after background re-alignment).
   * Preserves current offset.
   */
  updateLines(lines: AlignedLine[]): void {
    this.lines = [...lines].sort((a, b) => a.startTime - b.startTime);
  }

  // ── Private helpers ───────────────────────────────────────────────────────────

  private _autoReanchor(
    playbackTime: number,
    active:       AlignedLine,
    nowMs:        number,
  ): void {
    if (active.confidence < MIN_CONFIDENCE_ANCHOR) return;

    const expected = active.startTime + this.offset;
    const rawDrift = playbackTime - expected;

    if (Math.abs(rawDrift) < 0.05) { this.lastAnchorMs = nowMs; return; }

    this._applyCorrection(rawDrift, playbackTime, expected, "smooth");
    this.lastAnchorMs = nowMs;
  }

  private _applyCorrection(
    rawDrift:     number,
    playbackTime: number,
    expected:     number,
    type:         DriftEvent["type"],
  ): void {
    this.driftHistory.push(rawDrift);
    if (this.driftHistory.length > this.maxHistLen) this.driftHistory.shift();

    let correction: number;
    let eventType = type;

    if (Math.abs(rawDrift) > MAX_HARD_DRIFT) {
      // Hard reset — snap immediately
      correction = rawDrift;
      eventType  = "hard_reanchor";
    } else {
      // Exponential smoothing — gradual correction
      correction = rawDrift * SMOOTH_ALPHA;
    }

    this.offset += correction;

    this.driftLog.push({
      ts:            Date.now(),
      playbackTime,
      expectedTime:  expected,
      rawDrift,
      appliedOffset: this.offset,
      type:          eventType,
    });
  }
}

// ── Standalone helper ──────────────────────────────────────────────────────────

/**
 * Find the line index that should be active at `time`, with no drift correction.
 * Pure function — useful for SSR / non-class contexts.
 */
export function findActiveLineIndex(lines: AlignedLine[], time: number): number {
  for (let i = lines.length - 1; i >= 0; i--) {
    if (lines[i]!.startTime <= time) return i;
  }
  return -1;
}
