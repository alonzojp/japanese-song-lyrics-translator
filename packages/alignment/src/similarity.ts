/**
 * Text similarity functions for Japanese lyric matching.
 *
 * Three measures, combined into a weighted composite:
 *   levenshtein  — character edit distance (handles substitutions/typos)
 *   bigramOverlap — character bigram Jaccard (handles reordering/insertions)
 *   romajiSim    — romaji comparison (handles kana variant / kanji misread)
 */
import type { MatchOptions } from "./types.js";

const DEFAULT_LEV_W    = 0.50;
const DEFAULT_BIGRAM_W = 0.30;
const DEFAULT_ROMAJI_W = 0.20;

// ── Levenshtein ────────────────────────────────────────────────────────────────

/** Wagner-Fischer edit distance. */
export function levenshtein(a: string, b: string): number {
  if (a === b) return 0;
  if (!a.length) return b.length;
  if (!b.length) return a.length;

  const m = a.length;
  const n = b.length;
  // Use two-row rolling array to save memory for long strings
  let prev = Array.from({ length: n + 1 }, (_, j) => j);
  let curr = new Array<number>(n + 1);

  for (let i = 1; i <= m; i++) {
    curr[0] = i;
    for (let j = 1; j <= n; j++) {
      const sub = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(
        (prev[j]     ?? 0) + 1,    // deletion
        (curr[j - 1] ?? 0) + 1,    // insertion
        (prev[j - 1] ?? 0) + sub   // substitution
      );
    }
    [prev, curr] = [curr, prev];
  }
  return prev[n]!;
}

/** Levenshtein similarity: 1 - dist/maxLen. Range [0, 1]. */
export function levenshteinSim(a: string, b: string): number {
  if (!a && !b) return 1;
  const maxLen = Math.max(a.length, b.length);
  if (!maxLen) return 1;
  return 1 - levenshtein(a, b) / maxLen;
}

// ── Character bigram overlap ───────────────────────────────────────────────────

function getBigrams(s: string): Map<string, number> {
  const map = new Map<string, number>();
  for (let i = 0; i < s.length - 1; i++) {
    const bg = s[i]! + s[i + 1]!;
    map.set(bg, (map.get(bg) ?? 0) + 1);
  }
  return map;
}

function bigramIntersection(a: Map<string, number>, b: Map<string, number>): number {
  let count = 0;
  for (const [bg, ca] of a) {
    const cb = b.get(bg) ?? 0;
    count += Math.min(ca, cb);
  }
  return count;
}

/**
 * Dice coefficient on character bigrams.
 * Better than exact match for Japanese (no word separators).
 */
export function bigramSim(a: string, b: string): number {
  if (!a && !b) return 1;
  if (a.length < 2 || b.length < 2) return a === b ? 1 : 0;

  const ba = getBigrams(a);
  const bb = getBigrams(b);
  const intersect = bigramIntersection(ba, bb);
  const total     = [...ba.values()].reduce((s, v) => s + v, 0)
                  + [...bb.values()].reduce((s, v) => s + v, 0);
  return total > 0 ? (2 * intersect) / total : 0;
}

// ── Romaji similarity ──────────────────────────────────────────────────────────

/**
 * Levenshtein similarity on pre-computed romaji strings.
 * Handles kanji misreadings (e.g. 空 read as そら vs くう).
 */
export function romajiSim(ra: string, rb: string): number {
  return levenshteinSim(ra, rb);
}

// ── Composite ──────────────────────────────────────────────────────────────────

/**
 * Weighted composite similarity score.
 * `na` and `nb` are normalized texts; `ra` and `rb` are pre-computed romaji.
 */
export function computeSimilarity(
  na:   string,
  nb:   string,
  ra:   string,
  rb:   string,
  opts: Pick<MatchOptions, "levWeight" | "bigramWeight" | "romajiWeight"> = {}
): number {
  const wL = opts.levWeight    ?? DEFAULT_LEV_W;
  const wB = opts.bigramWeight ?? DEFAULT_BIGRAM_W;
  const wR = opts.romajiWeight ?? DEFAULT_ROMAJI_W;

  const lev    = levenshteinSim(na, nb);
  const bigram = bigramSim(na, nb);
  const rom    = romajiSim(ra, rb);

  return wL * lev + wB * bigram + wR * rom;
}

/**
 * Build a full [official × transcript] similarity matrix.
 * Returns a 2D array where matrix[i][j] = similarity(official[i], transcript[j]).
 */
export function buildSimilarityMatrix(
  normOfficial:   string[],
  normTranscript: string[],
  romajiOfficial:   string[],
  romajiTranscript: string[],
  opts: MatchOptions = {}
): number[][] {
  return normOfficial.map((no, i) =>
    normTranscript.map((nt, j) =>
      computeSimilarity(no, nt, romajiOfficial[i]!, romajiTranscript[j]!, opts)
    )
  );
}

/** Token overlap: what fraction of unique characters in `a` also appear in `b`. */
export function tokenOverlap(a: string, b: string): number {
  if (!a || !b) return 0;
  const setA = new Set([...a]);
  const setB = new Set([...b]);
  let shared = 0;
  for (const c of setA) if (setB.has(c)) shared++;
  return shared / setA.size;
}
