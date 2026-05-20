/**
 * Chorus (repeated section) detection.
 *
 * Problem: A song's chorus repeats 2-4 times with identical lyrics.
 * DTW on the full matrix would greedily assign all occurrences to the
 * first transcript match. This module detects repeated blocks beforehand
 * so the matcher can enforce temporal ordering.
 *
 * Algorithm:
 *   1. Build a self-similarity matrix on the normalized official lyrics.
 *   2. Find off-diagonal stripes of high similarity (sliding window).
 *   3. Cluster adjacent matches into ChorusGroup objects.
 *   4. Each ChorusGroup knows its lineIndices + occurrenceCount.
 */
import { bigramSim } from "./similarity.js";
import { normalize } from "./normalizer.js";
import type { ChorusGroup } from "./types.js";

const DEFAULT_THRESHOLD = 0.78;
const DEFAULT_MIN_LINES = 2;
const DEFAULT_MAX_GAP   = 3;   // max lines between similar blocks to consider same group

// ── Self-similarity ────────────────────────────────────────────────────────────

function selfSimilarityMatrix(normLines: string[]): number[][] {
  const n = normLines.length;
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) =>
      i === j ? 1 : bigramSim(normLines[i]!, normLines[j]!)
    )
  );
}

// ── Block detection ────────────────────────────────────────────────────────────

interface SimilarBlock {
  startA: number;
  startB: number;
  length: number;
  avgSim: number;
}

function findSimilarBlocks(
  simMatrix: number[][],
  threshold: number,
  minLength: number,
): SimilarBlock[] {
  const n = simMatrix.length;
  const blocks: SimilarBlock[] = [];

  // Scan diagonals offset from the main diagonal
  for (let offset = minLength; offset < n; offset++) {
    let runStart  = -1;
    let runSimSum = 0;
    let runLen    = 0;

    for (let i = 0; i + offset < n; i++) {
      const j   = i + offset;
      const sim = simMatrix[i]![j]!;

      if (sim >= threshold) {
        if (runStart === -1) { runStart = i; runSimSum = 0; runLen = 0; }
        runSimSum += sim;
        runLen++;
      } else {
        if (runStart !== -1 && runLen >= minLength) {
          blocks.push({
            startA:  runStart,
            startB:  runStart + offset,
            length:  runLen,
            avgSim:  runSimSum / runLen,
          });
        }
        runStart = -1;
        runLen   = 0;
      }
    }
    if (runStart !== -1 && runLen >= minLength) {
      blocks.push({
        startA: runStart,
        startB: runStart + offset,
        length: runLen,
        avgSim: runSimSum / runLen,
      });
    }
  }
  return blocks;
}

// ── Group merging ──────────────────────────────────────────────────────────────

/**
 * Merge overlapping / adjacent blocks that reference the same underlying range.
 * Returns canonical groups where `lineIndices` covers the first occurrence.
 */
function mergeBlocks(
  blocks: SimilarBlock[],
  maxGap: number,
): Array<{ firstStart: number; offsets: number[]; length: number; avgSim: number }> {
  if (!blocks.length) return [];

  const sorted = [...blocks].sort((a, b) => a.startA - b.startA || a.startB - b.startB);
  const groups: Map<number, { offsets: Set<number>; length: number; simSum: number; count: number }> = new Map();

  for (const blk of sorted) {
    let merged = false;
    for (const [key, grp] of groups) {
      if (
        Math.abs(blk.startA - key) <= maxGap &&
        Math.abs(blk.length - grp.length) <= 1
      ) {
        grp.offsets.add(blk.startB - key);
        grp.simSum += blk.avgSim;
        grp.count++;
        merged = true;
        break;
      }
    }
    if (!merged) {
      groups.set(blk.startA, {
        offsets: new Set([blk.startB - blk.startA]),
        length:  blk.length,
        simSum:  blk.avgSim,
        count:   1,
      });
    }
  }

  return [...groups.entries()].map(([firstStart, grp]) => ({
    firstStart,
    offsets:  [...grp.offsets].sort((a, b) => a - b),
    length:   grp.length,
    avgSim:   grp.simSum / grp.count,
  }));
}

// ── Public API ─────────────────────────────────────────────────────────────────

/**
 * Detect repeated sections (choruses) in a list of official lyric lines.
 * Returns ChorusGroup objects, one per unique repeated block.
 */
export function detectChoruses(
  lines:     string[],
  threshold  = DEFAULT_THRESHOLD,
  minLines   = DEFAULT_MIN_LINES,
  maxGap     = DEFAULT_MAX_GAP,
): ChorusGroup[] {
  if (lines.length < minLines * 2) return [];

  const normalized = lines.map(normalize);
  const simMatrix  = selfSimilarityMatrix(normalized);
  const blocks     = findSimilarBlocks(simMatrix, threshold, minLines);
  const merged     = mergeBlocks(blocks, maxGap);

  return merged.map((grp, id) => {
    const firstStart   = grp.firstStart;
    const lineIndices  = Array.from({ length: grp.length }, (_, k) => firstStart + k);
    // All occurrence start positions: firstStart + each offset
    const allStarts    = [firstStart, ...grp.offsets.map((o) => firstStart + o)];

    return {
      id,
      lineIndices,
      occurrences:    allStarts.length,
      firstLineIndex: firstStart,
      groupSize:      grp.length,
      avgSimilarity:  grp.avgSim,
    };
  });
}

/**
 * Given a list of ChorusGroup objects and a transcript time, determine
 * which occurrence of the chorus we are most likely in.
 *
 * Uses the expected time of each occurrence (evenly spaced across the song)
 * plus a provided set of candidate transcript timestamps to disambiguate.
 *
 * Returns the 0-based occurrence index.
 */
export function chorusOccurrenceForTime(
  group:                  ChorusGroup,
  transcriptDuration:     number,
  officialLineCount:      number,
  currentTranscriptIndex: number,
  transcriptLength:       number,
): number {
  if (group.occurrences <= 1) return 0;

  // Approximate when each occurrence starts in the transcript
  const relativeStart = group.firstLineIndex / officialLineCount;
  const groupRelSize  = group.groupSize      / officialLineCount;

  // Expected transcript indices for each occurrence
  const expectedIndices = Array.from({ length: group.occurrences }, (_, k) => {
    const relPos = relativeStart + k * (1 - relativeStart - groupRelSize) / Math.max(1, group.occurrences - 1);
    return Math.round(relPos * transcriptLength);
  });

  // Find the occurrence whose expected index is closest to currentTranscriptIndex
  let best = 0;
  let bestDist = Infinity;
  for (let k = 0; k < expectedIndices.length; k++) {
    const dist = Math.abs((expectedIndices[k] ?? 0) - currentTranscriptIndex);
    if (dist < bestDist) { bestDist = dist; best = k; }
  }
  return best;
}
