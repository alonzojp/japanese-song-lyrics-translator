import type { Token, AnalysisResult } from "./token.js";

// ── Primary karaoke types (as specified in product requirements) ───────────────

export interface WordTiming {
  word:  string;
  start: number;   // seconds
  end:   number;   // seconds
}

/** Primary display unit for the karaoke player. */
export interface LyricLine {
  id:         string;        // "line-0", "line-1" …
  text:       string;
  startTime:  number;
  endTime:    number;
  words?:     WordTiming[];
  /** 0.0–1.0 alignment confidence for this specific line */
  confidence?: number;
}

// ── Alignment metadata ────────────────────────────────────────────────────────

export interface AlignmentConfidence {
  overall:          number;
  wordCoverage:     number;
  avgWordScore:     number;
  lowScoreWords:    number;
  gapAnomalies:     number[];
  flaggedLines:     number[];
  method:           string;
  lineConfidences:  number[];
  notes:            string[];
}

export interface AlignmentMeta {
  youtubeId:              string;
  transcriptionBackend:   string;
  alignmentMethod:        string;
  whisperModel:           string;
  audioDetails: {
    duration:   number;
    language:   string;
    vocalsOnly: boolean;
  };
  confidence:  AlignmentConfidence;
  processedAt: string;
}

// ── DB-layer lyric line (stored in SQLite) ────────────────────────────────────

export interface StoredLyricLine {
  index:    number;
  startTime: number | null;
  endTime:   number | null;
  japanese:  string;
  words:     WordTiming[] | null;
  tokens:    Token[] | null;
  analysis:  AnalysisResult | null;
}

// ── Song ──────────────────────────────────────────────────────────────────────

export interface Song {
  id:         string;
  youtubeUrl: string;
  youtubeId:  string;
  title:      string | null;
  artist:     string | null;
  thumbnail:  string | null;
  lyrics:     StoredLyricLine[];
  createdAt:  Date;
  updatedAt:  Date;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

export function extractYoutubeId(url: string): string | null {
  const patterns = [
    /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/,
    /youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})/,
  ];
  for (const pattern of patterns) {
    const match = url.match(pattern);
    if (match?.[1]) return match[1];
  }
  return null;
}

/** Convert a stored DB line to a karaoke LyricLine. */
export function storedToLyricLine(stored: StoredLyricLine): LyricLine {
  return {
    id:         `line-${stored.index}`,
    text:       stored.japanese,
    startTime:  stored.startTime ?? 0,
    endTime:    stored.endTime   ?? 0,
    words:      stored.words ?? undefined,
  };
}
