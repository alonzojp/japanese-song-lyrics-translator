// ── Lyrics Provider Contract ───────────────────────────────────────────────────
// Mirrors services/worker/lyrics/types.py — keep in sync.

export interface LyricsLine {
  text:      string;
  startTime: number | null;
  endTime:   number | null;
}

export interface ProviderAttempt {
  provider:      string;
  priority:      number;
  succeeded:     boolean;
  failureReason: string | null;
  durationMs:    number;
  debugNotes:    string[];
}

export interface LyricsResult {
  youtubeId:        string;
  provider:         string;
  /** 0.0–1.0 — how much we trust this source */
  confidence:       number;
  hasTimestamps:    boolean;
  /** 0.0–1.0 — fraction of lines that have timing data */
  timestampQuality: number;
  sourceUrl:        string | null;
  lines:            LyricsLine[];
  rawText:          string;
  metadata:         Record<string, unknown>;
  providerAttempts: ProviderAttempt[];
  cachedAt:         string;
}

// ── Request shapes ─────────────────────────────────────────────────────────────

export interface FetchLyricsRequest {
  youtubeId:   string;
  youtubeUrl:  string;
  title?:      string;
  uploader?:   string;
  description?: string;
  duration?:   number;
  /** Force re-fetch even if cached */
  force?:      boolean;
}

export interface ManualLyricsRequest {
  youtubeId: string;
  text:      string;
}

// ── Provider debug visualization helpers ──────────────────────────────────────

export type ConfidenceTier = "high" | "medium" | "low" | "none";

export function getConfidenceTier(confidence: number): ConfidenceTier {
  if (confidence >= 0.80) return "high";
  if (confidence >= 0.55) return "medium";
  if (confidence > 0)     return "low";
  return "none";
}

export const PROVIDER_LABELS: Record<string, string> = {
  youtube_captions:      "YouTube (official captions)",
  youtube_auto_captions: "YouTube (auto-generated)",
  description:           "YouTube description",
  comments:              "YouTube comments",
  genius:                "Genius",
  utaten:                "Utaten",
  petitlyrics:           "PetitLyrics",
  manual:                "Manual upload",
  none:                  "No source",
};
