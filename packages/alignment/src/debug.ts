/**
 * Alignment debugging utilities.
 *
 * All methods return plain data structures (no DOM / React deps)
 * so they can run server-side or be serialized to JSON.
 *
 * Consumers:
 *   - Transcript-vs-lyric diff viewer
 *   - Per-line confidence bar chart
 *   - Drift time-series graph
 *   - Similarity heatmap
 */
import type { AlignedLine, DriftAnchor, MatchResult, MatchStats } from "./types.js";
import type { TranscriptSegment, OfficialLine } from "./types.js";
import { normalize } from "./normalizer.js";
import { levenshteinSim, bigramSim } from "./similarity.js";

// ── Diff viewer ────────────────────────────────────────────────────────────────

export type DiffOp = "equal" | "replace" | "insert" | "delete";

export interface DiffToken {
  op:             DiffOp;
  officialChar?:  string;
  transcriptChar?: string;
}

export interface LineDiff {
  lineId:          string;
  officialText:    string;
  transcriptText:  string;
  similarity:      number;
  confidence:      number;
  tokens:          DiffToken[];     // character-level diff
  isChorus:        boolean;
  anchor:          boolean;
}

/** Build a character-level diff between two strings (LCS-based). */
function charDiff(a: string, b: string): DiffToken[] {
  const na = normalize(a);
  const nb = normalize(b);
  const m = na.length;
  const n = nb.length;

  // LCS DP
  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i]![j] = na[i - 1] === nb[j - 1]
        ? dp[i - 1]![j - 1]! + 1
        : Math.max(dp[i - 1]![j]!, dp[i]![j - 1]!);
    }
  }

  // Traceback
  const tokens: DiffToken[] = [];
  let i = m, j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && na[i - 1] === nb[j - 1]) {
      tokens.push({ op: "equal", officialChar: na[i - 1], transcriptChar: nb[j - 1] });
      i--; j--;
    } else if (i > 0 && (j === 0 || dp[i - 1]![j]! >= dp[i]![j - 1]!)) {
      tokens.push({ op: "delete", officialChar: na[i - 1] });
      i--;
    } else {
      tokens.push({ op: "insert", transcriptChar: nb[j - 1] });
      j--;
    }
  }
  return tokens.reverse();
}

// ── Heatmap data ───────────────────────────────────────────────────────────────

export interface HeatmapData {
  /** officialCount × transcriptCount similarity values */
  matrix:          number[][];
  officialLabels:  string[];
  transcriptLabels: string[];
  /** Highlighted cells from the DTW path */
  pathCells:       Array<[number, number]>;
  /** Anchor cells */
  anchorCells:     Array<[number, number]>;
}

// ── Confidence series ──────────────────────────────────────────────────────────

export interface ConfidencePoint {
  lineIndex:   number;
  lineId:      string;
  confidence:  number;
  similarity:  number;
  isChorus:    boolean;
  isFlagged:   boolean;
  isAnchor:    boolean;
}

// ── Drift series ───────────────────────────────────────────────────────────────

export interface DriftPoint {
  lineIndex: number;
  lineId:    string;
  startTime: number;
  drift:     number;
  isAnchor:  boolean;
}

// ── Main debugger class ────────────────────────────────────────────────────────

export class AlignmentDebugger {
  constructor(
    private readonly result:     MatchResult,
    private readonly transcript: TranscriptSegment[],
    private readonly official:   OfficialLine[],
  ) {}

  // ── Transcript diff viewer ─────────────────────────────────────────────────

  toTranscriptDiff(): LineDiff[] {
    const anchorSet = new Set(this.result.anchors.map((a) => a.officialIndex));

    return this.result.lines.map((line, i) => ({
      lineId:         line.id,
      officialText:   line.text,
      transcriptText: line.transcriptText,
      similarity:     line.textSimilarity,
      confidence:     line.confidence,
      tokens:         charDiff(line.text, line.transcriptText),
      isChorus:       line.isChorus,
      anchor:         anchorSet.has(i),
    }));
  }

  // ── Confidence series ──────────────────────────────────────────────────────

  toConfidenceSeries(): ConfidencePoint[] {
    const anchors   = new Set(this.result.anchors.map((a) => a.officialIndex));
    const flagged   = new Set(this.result.stats.flaggedLines);
    return this.result.lines.map((line, i) => ({
      lineIndex:  i,
      lineId:     line.id,
      confidence: line.confidence,
      similarity: line.textSimilarity,
      isChorus:   line.isChorus,
      isFlagged:  flagged.has(i),
      isAnchor:   anchors.has(i),
    }));
  }

  // ── Drift series ───────────────────────────────────────────────────────────

  toDriftSeries(): DriftPoint[] {
    const anchors = new Set(this.result.anchors.map((a) => a.officialIndex));
    return this.result.lines.map((line, i) => ({
      lineIndex: i,
      lineId:    line.id,
      startTime: line.startTime,
      drift:     line.drift,
      isAnchor:  anchors.has(i),
    }));
  }

  // ── Similarity heatmap (sampled for large songs) ───────────────────────────

  toHeatmapData(maxDim = 50): HeatmapData {
    const n = this.official.length;
    const m = this.transcript.length;

    // Sample rows/cols if too large
    const rowStep = Math.max(1, Math.ceil(n / maxDim));
    const colStep = Math.max(1, Math.ceil(m / maxDim));

    const rowIdxs = Array.from({ length: Math.ceil(n / rowStep) }, (_, i) => i * rowStep);
    const colIdxs = Array.from({ length: Math.ceil(m / colStep) }, (_, j) => j * colStep);

    const normOff = rowIdxs.map((i) => normalize(this.official[i]?.text ?? ""));
    const normTx  = colIdxs.map((j) => normalize(this.transcript[j]?.text ?? ""));

    const matrix = normOff.map((a) => normTx.map((b) => parseFloat(bigramSim(a, b).toFixed(3))));

    // Map DTW path cells to sampled coordinates
    const pathCells = this.result.lines
      .filter((l) => l.transcriptIndex >= 0)
      .map((l): [number, number] => {
        const sampledRow = Math.floor(rowIdxs.findIndex((i) => i >= this.official.findIndex((o) => o.text === l.text)));
        const sampledCol = Math.floor(colIdxs.findIndex((j) => j >= l.transcriptIndex));
        return [Math.max(0, sampledRow), Math.max(0, sampledCol)];
      });

    const anchorCells = this.result.anchors
      .map((a): [number, number] => {
        const r = Math.floor(a.officialIndex   / rowStep);
        const c = Math.floor(a.transcriptIndex / colStep);
        return [r, c];
      });

    return {
      matrix,
      officialLabels:   rowIdxs.map((i) => this.official[i]?.text.slice(0, 12) ?? ""),
      transcriptLabels: colIdxs.map((j) => this.transcript[j]?.text.slice(0, 12) ?? ""),
      pathCells,
      anchorCells,
    };
  }

  // ── Summary ────────────────────────────────────────────────────────────────

  toSummary(): Record<string, unknown> {
    const s = this.result.stats;
    return {
      officialLines:    this.official.length,
      transcriptSegs:   this.transcript.length,
      avgConfidence:    s.avgConfidence,
      avgSimilarity:    s.avgSimilarity,
      chorusGroups:     s.chorusGroups,
      anchors:          s.totalAnchors,
      flaggedLines:     s.flaggedLines.length,
      unmatched:        s.unmatched.length,
      driftRange:       `${s.minDrift.toFixed(2)}s – ${s.maxDrift.toFixed(2)}s`,
    };
  }
}
