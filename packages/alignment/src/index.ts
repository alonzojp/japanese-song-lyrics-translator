// ── Types ──────────────────────────────────────────────────────────────────────
export type {
  TranscriptSegment,
  OfficialLine,
  AlignedLine,
  ChorusGroup,
  DriftAnchor,
  RawMatch,
  MatchOptions,
  MatchResult,
  MatchStats,
} from "./types.js";

// ── Text processing ────────────────────────────────────────────────────────────
export {
  normalize,
  normalizeSoft,
  toNormalizedRomaji,
  kataToHira,
  hasJapanese,
  japaneseRatio,
  normalizeAll,
} from "./normalizer.js";

// ── Similarity ─────────────────────────────────────────────────────────────────
export {
  levenshtein,
  levenshteinSim,
  bigramSim,
  romajiSim,
  computeSimilarity,
  buildSimilarityMatrix,
  tokenOverlap,
} from "./similarity.js";

// ── DTW ────────────────────────────────────────────────────────────────────────
export {
  dtw,
  segmentedDTW,
  resolvePathToMapping,
} from "./dtw.js";
export type { DTWResult } from "./dtw.js";

// ── Chorus detection ───────────────────────────────────────────────────────────
export {
  detectChoruses,
  chorusOccurrenceForTime,
} from "./chorus-detector.js";

// ── Main matcher ───────────────────────────────────────────────────────────────
export { matchLyricLines } from "./matcher.js";

// ── Real-time sync ─────────────────────────────────────────────────────────────
export { SyncEngine, findActiveLineIndex } from "./sync-engine.js";
export type { DriftEvent, SyncState } from "./sync-engine.js";

// ── Debug ──────────────────────────────────────────────────────────────────────
export { AlignmentDebugger } from "./debug.js";
export type {
  DiffToken,
  LineDiff,
  HeatmapData,
  ConfidencePoint,
  DriftPoint,
} from "./debug.js";
