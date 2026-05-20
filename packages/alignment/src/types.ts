import type { WordTiming } from "@japanese-lyrics/shared";

// ── Input types ────────────────────────────────────────────────────────────────

/** One segment from WhisperX — has timestamps but possibly imperfect text. */
export interface TranscriptSegment {
  text:      string;
  startTime: number;
  endTime:   number;
  words?:    WordTiming[];
}

/** One line from a lyrics provider — correct text, no timestamps. */
export interface OfficialLine {
  index: number;
  text:  string;
}

// ── Output types ───────────────────────────────────────────────────────────────

/** An official lyric line with timestamps derived from the transcript match. */
export interface AlignedLine {
  id:              string;       // "line-0", "line-1" …
  text:            string;       // official lyric text (authoritative)
  transcriptText:  string;       // what Whisper actually said
  startTime:       number;
  endTime:         number;
  words?:          WordTiming[];

  // ── Match quality ────────────────────────────────────────────
  confidence:      number;       // 0.0–1.0 composite
  textSimilarity:  number;       // 0.0–1.0 normalized vs transcript
  temporalScore:   number;       // 0.0–1.0 how plausible the timestamp is
  neighborScore:   number;       // 0.0–1.0 consistency with adjacent matches

  // ── Debug / metadata ─────────────────────────────────────────
  transcriptIndex: number;       // which WhisperX segment was matched
  isChorus:        boolean;
  chorusGroupId?:  number;       // which chorus group (0-based)
  chorusOccurrence?: number;     // 0 = first time, 1 = second time, etc.
  drift:           number;       // applied time-offset correction (seconds)
  anchorDistance:  number;       // lines from nearest anchor point
}

// ── Intermediate types ─────────────────────────────────────────────────────────

export interface ChorusGroup {
  id:               number;
  lineIndices:      number[];     // official lyric indices in this group
  occurrences:      number;       // how many times the chorus repeats
  firstLineIndex:   number;       // index of first line in first occurrence
  groupSize:        number;       // number of lines per occurrence
  avgSimilarity:    number;       // internal similarity within group
}

export interface DriftAnchor {
  officialIndex:   number;
  transcriptIndex: number;
  similarity:      number;
  expectedTime:    number;
  actualTime:      number;
  correctedOffset: number;
}

export interface RawMatch {
  officialIndex:   number;
  transcriptIndex: number;
  similarity:      number;
  startTime:       number;
  endTime:         number;
  isChorus:        boolean;
}

// ── Match options ──────────────────────────────────────────────────────────────

export interface MatchOptions {
  /** DTW Sakoe-Chiba band width (default: 15% of transcript length, min 8) */
  dtwWindow?:            number;
  /** Minimum similarity to consider a match valid (default: 0.20) */
  minSimilarity?:        number;
  /** Minimum similarity to use a match as a drift anchor (default: 0.65) */
  anchorThreshold?:      number;
  /** Similarity multiplier weight for levenshtein (default: 0.50) */
  levWeight?:            number;
  /** Similarity multiplier weight for bigram overlap (default: 0.30) */
  bigramWeight?:         number;
  /** Similarity multiplier weight for romaji (default: 0.20) */
  romajiWeight?:         number;
  /** Minimum chorus group size in lines (default: 2) */
  minChorusLines?:       number;
  /** Minimum similarity between repeated sections to count as chorus (default: 0.78) */
  chorusThreshold?:      number;
  /** Apply drift correction between anchors (default: true) */
  driftCorrection?:      boolean;
  /** Maximum drift to tolerate before forced re-anchor (seconds, default: 3.0) */
  maxDrift?:             number;
}

// ── Result ─────────────────────────────────────────────────────────────────────

export interface MatchStats {
  avgConfidence:    number;
  avgSimilarity:    number;
  chorusGroups:     number;
  totalAnchors:     number;
  maxDrift:         number;
  minDrift:         number;
  flaggedLines:     number[];   // indices of low-confidence lines
  unmatched:        number[];   // official indices with no good match
}

export interface MatchResult {
  lines:  AlignedLine[];
  stats:  MatchStats;
  anchors: DriftAnchor[];
}
