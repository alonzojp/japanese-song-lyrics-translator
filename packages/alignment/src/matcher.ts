/**
 * matchLyricLines — the core synchronization engine.
 *
 * Takes:
 *   transcriptLines  WhisperX segments (timestamped, imperfect text)
 *   officialLyrics   Provider lyrics (correct text, no timestamps)
 *
 * Returns:
 *   AlignedLine[]    Official lyrics with timestamps from transcript + confidence
 *
 * Pipeline:
 *   1. Normalize both sequences
 *   2. Detect repeated choruses in official lyrics
 *   3. Build similarity matrix [official × transcript]
 *   4. DTW alignment (segmented for long songs)
 *   5. Resolve path → per-line transcript index
 *   6. Chorus disambiguation (temporal context)
 *   7. Future-window re-scoring for uncertain matches
 *   8. Anchor-based drift correction
 *   9. Confidence scoring
 */
import { normalizeAll } from "./normalizer.js";
import { buildSimilarityMatrix, computeSimilarity, bigramSim } from "./similarity.js";
import { dtw, segmentedDTW, resolvePathToMapping } from "./dtw.js";
import { detectChoruses, chorusOccurrenceForTime } from "./chorus-detector.js";
import type {
  AlignedLine,
  ChorusGroup,
  DriftAnchor,
  MatchOptions,
  MatchResult,
  MatchStats,
  OfficialLine,
  RawMatch,
  TranscriptSegment,
} from "./types.js";

// ── Constants ──────────────────────────────────────────────────────────────────

const DEFAULT_OPTIONS: Required<MatchOptions> = {
  dtwWindow:       0,        // 0 = auto (15% of transcript length, min 8)
  minSimilarity:   0.20,
  anchorThreshold: 0.62,
  levWeight:       0.50,
  bigramWeight:    0.30,
  romajiWeight:    0.20,
  minChorusLines:  2,
  chorusThreshold: 0.78,
  driftCorrection: true,
  maxDrift:        3.0,
};

const FUTURE_WINDOW_SIZE = 5;     // look-ahead for re-scoring uncertain matches
const LONG_SONG_THRESHOLD = 40;   // use segmented DTW above this many official lines

// ── Helpers ────────────────────────────────────────────────────────────────────

function opt(options: MatchOptions): Required<MatchOptions> {
  return { ...DEFAULT_OPTIONS, ...options };
}

/** Choose DTW window: explicit > auto-calculated. */
function dtwWindow(opts: Required<MatchOptions>, transcriptLen: number): number {
  if (opts.dtwWindow > 0) return opts.dtwWindow;
  return Math.max(8, Math.floor(transcriptLen * 0.15));
}

// ── Step 5: path → mapping ─────────────────────────────────────────────────────

function buildRawMatches(
  mapping:    number[],
  official:   OfficialLine[],
  transcript: TranscriptSegment[],
  simMatrix:  number[][],
  chorusMap:  Map<number, ChorusGroup>,
): RawMatch[] {
  return official.map((off, i) => {
    const j = mapping[i] ?? -1;
    if (j < 0 || j >= transcript.length) {
      return {
        officialIndex:   i,
        transcriptIndex: -1,
        similarity:      0,
        startTime:       0,
        endTime:         0,
        isChorus:        chorusMap.has(i),
      };
    }
    const seg = transcript[j]!;
    return {
      officialIndex:   i,
      transcriptIndex: j,
      similarity:      simMatrix[i]?.[j] ?? 0,
      startTime:       seg.startTime,
      endTime:         seg.endTime,
      isChorus:        chorusMap.has(i),
    };
  });
}

// ── Step 6: chorus disambiguation ─────────────────────────────────────────────

function disambiguateChoruses(
  raw:        RawMatch[],
  groups:     ChorusGroup[],
  transcript: TranscriptSegment[],
): RawMatch[] {
  if (!groups.length) return raw;

  const result = [...raw];

  for (const group of groups) {
    const { lineIndices, occurrences, groupSize } = group;
    if (occurrences < 2) continue;

    // Find all transcript segment ranges that are plausible for each occurrence
    // Use the non-chorus lines before/after as context anchors
    const allOccurrenceStarts: number[] = [];
    for (let k = 0; k < occurrences; k++) {
      const firstInOccurrence = lineIndices[0]! + k * groupSize;
      if (firstInOccurrence < raw.length) {
        allOccurrenceStarts.push(raw[firstInOccurrence]?.transcriptIndex ?? -1);
      }
    }

    // Sort occurrences by transcript index (temporal order)
    const sortedStarts = allOccurrenceStarts
      .filter((s) => s >= 0)
      .sort((a, b) => a - b);

    // Re-assign each occurrence's lines to their sorted cluster
    for (let k = 0; k < Math.min(occurrences, sortedStarts.length); k++) {
      const clusterCenter = sortedStarts[k]!;
      const searchRadius  = Math.ceil(groupSize * 1.5);
      const lineStart     = lineIndices[0]! + k * groupSize;

      for (let li = 0; li < groupSize; li++) {
        const officialIdx = lineStart + li;
        if (officialIdx >= result.length) break;

        // Find best matching transcript segment near clusterCenter + li
        const targetJ   = clusterCenter + li;
        const windowLo  = Math.max(0, targetJ - searchRadius);
        const windowHi  = Math.min(transcript.length - 1, targetJ + searchRadius);

        let bestJ     = targetJ;
        let bestScore = -1;
        for (let j = windowLo; j <= windowHi; j++) {
          // Weight: prefer segments near the expected temporal position
          const proximityBonus = 1 - Math.abs(j - targetJ) / (searchRadius + 1) * 0.3;
          const sim = (result[officialIdx]?.similarity ?? 0) * proximityBonus;
          if (sim > bestScore) { bestScore = sim; bestJ = j; }
        }

        const seg = transcript[bestJ];
        if (seg) {
          result[officialIdx] = {
            ...result[officialIdx]!,
            transcriptIndex: bestJ,
            startTime:       seg.startTime,
            endTime:         seg.endTime,
            isChorus:        true,
          };
        }
      }
    }
  }

  return result;
}

// ── Step 7: future-window re-scoring ──────────────────────────────────────────

/**
 * For lines with low similarity, look FUTURE_WINDOW_SIZE segments ahead
 * in the transcript for a better match. Re-assigns if significantly better.
 */
function futureWindowRescore(
  raw:        RawMatch[],
  transcript: TranscriptSegment[],
  simMatrix:  number[][],
  opts:       Required<MatchOptions>,
): RawMatch[] {
  return raw.map((m, i) => {
    if (m.transcriptIndex < 0) return m;
    if (m.similarity >= opts.anchorThreshold) return m;   // already confident

    const lo = m.transcriptIndex;
    const hi = Math.min(transcript.length - 1, lo + FUTURE_WINDOW_SIZE);

    let bestJ   = m.transcriptIndex;
    let bestSim = m.similarity;
    for (let j = lo; j <= hi; j++) {
      const s = simMatrix[i]?.[j] ?? 0;
      if (s > bestSim + 0.05) {   // must be meaningfully better to switch
        bestSim = s;
        bestJ   = j;
      }
    }

    if (bestJ === m.transcriptIndex) return m;
    const seg = transcript[bestJ]!;
    return { ...m, transcriptIndex: bestJ, similarity: bestSim, startTime: seg.startTime, endTime: seg.endTime };
  });
}

// ── Step 8: drift correction ───────────────────────────────────────────────────

/**
 * Identify high-confidence matches as anchors.
 * Between anchors, apply linear offset interpolation to correct drift.
 *
 * "Drift" here means the progressive deviation of matched timestamps
 * from what they would be if the alignment were perfect. Without correction,
 * a small systematic error (e.g. WhisperX running 0.2s fast) accumulates.
 */
function applyDriftCorrection(
  raw:   RawMatch[],
  opts:  Required<MatchOptions>,
): { corrected: RawMatch[]; anchors: DriftAnchor[] } {
  if (!opts.driftCorrection) return { corrected: raw, anchors: [] };

  // Collect anchor points: high-sim, monotone-increasing timestamps
  const anchors: DriftAnchor[] = [];
  let prevTime = -Infinity;

  for (const m of raw) {
    if (m.transcriptIndex < 0) continue;
    if (m.similarity >= opts.anchorThreshold && m.startTime > prevTime) {
      // Expected time based on linear interpolation over the whole transcript
      const expectedTime = m.startTime;  // for now, anchors are self-consistent
      anchors.push({
        officialIndex:   m.officialIndex,
        transcriptIndex: m.transcriptIndex,
        similarity:      m.similarity,
        expectedTime,
        actualTime:      m.startTime,
        correctedOffset: 0,   // filled below
      });
      prevTime = m.startTime;
    }
  }

  if (anchors.length < 2) return { corrected: raw, anchors };

  // Between consecutive anchors, compute per-anchor offset
  // For now the offset is 0 (anchors match their own transcript times).
  // In a live-sync scenario, this would be updated dynamically.
  // Here we use it to detect segments where drift is EXPECTED to be larger
  // (e.g. between anchor[k] and anchor[k+1] when the gap is large) and
  // flag those matches.

  const maxExpectedDrift = opts.maxDrift;
  const anchorSet        = new Set(anchors.map((a) => a.officialIndex));

  const corrected = raw.map((m) => {
    if (m.transcriptIndex < 0) return m;

    // Find nearest anchors before and after
    const prevAnchor = anchors.filter((a) => a.officialIndex < m.officialIndex).at(-1);
    const nextAnchor = anchors.find(   (a) => a.officialIndex > m.officialIndex);

    if (!prevAnchor && !nextAnchor) return m;

    // If match is between two good anchors and timestamp is monotone, it's fine
    const lo = prevAnchor?.actualTime ?? 0;
    const hi = nextAnchor?.actualTime ?? Infinity;

    // Flag if timestamp is outside the [prevAnchor, nextAnchor] range by maxDrift
    const flagged = m.startTime < lo - maxExpectedDrift || m.startTime > hi + maxExpectedDrift;
    if (!flagged) return m;

    // Correct: interpolate between anchors
    if (prevAnchor && nextAnchor) {
      const totalOff    = nextAnchor.officialIndex - prevAnchor.officialIndex;
      const relativeOff = m.officialIndex         - prevAnchor.officialIndex;
      const t = totalOff > 0 ? relativeOff / totalOff : 0;
      const interpolatedTime = prevAnchor.actualTime + t * (nextAnchor.actualTime - prevAnchor.actualTime);
      const diff = interpolatedTime - m.startTime;
      return {
        ...m,
        startTime: m.startTime + diff,
        endTime:   m.endTime   + diff,
      };
    }

    return m;
  });

  return { corrected, anchors };
}

// ── Step 9: build AlignedLine + confidence ─────────────────────────────────────

function buildAlignedLines(
  raw:         RawMatch[],
  official:    OfficialLine[],
  transcript:  TranscriptSegment[],
  chorusMap:   Map<number, ChorusGroup>,
  anchors:     DriftAnchor[],
): AlignedLine[] {
  const anchorSet    = new Set(anchors.map((a) => a.officialIndex));
  const anchorByIdx  = new Map(anchors.map((a) => [a.officialIndex, a]));
  let   nearestAnchorDist = Infinity;

  return raw.map((m, i) => {
    const offLine   = official[i]!;
    const seg       = m.transcriptIndex >= 0 ? transcript[m.transcriptIndex] : null;

    // Anchor distance (how far from nearest high-confidence anchor)
    for (const a of anchors) {
      const d = Math.abs(a.officialIndex - i);
      if (d < nearestAnchorDist) nearestAnchorDist = d;
    }

    // Temporal score: is the timestamp monotone and plausible?
    const prevTime  = i > 0 ? (raw[i - 1]?.startTime ?? 0) : 0;
    const nextTime  = i < raw.length - 1 ? (raw[i + 1]?.startTime ?? Infinity) : Infinity;
    const temporal  = m.startTime >= prevTime - 0.1 && m.startTime <= nextTime + 1.0
                      ? 1.0 : 0.3;

    // Neighbor score: do adjacent matches agree?
    const prevSim   = i > 0       ? (raw[i - 1]?.similarity ?? 0) : m.similarity;
    const nextSim   = i < raw.length - 1 ? (raw[i + 1]?.similarity ?? 0) : m.similarity;
    const neighbor  = (prevSim + m.similarity + nextSim) / 3;

    // Composite confidence
    const anchorBonus = anchorSet.has(i) ? 0.1 : 0;
    const confidence  = Math.min(1,
      0.50 * m.similarity +
      0.25 * temporal     +
      0.20 * neighbor     +
      0.05 * anchorBonus
    );

    const chorus = chorusMap.get(i);

    return {
      id:               `line-${i}`,
      text:             offLine.text,
      transcriptText:   seg?.text ?? "",
      startTime:        m.startTime,
      endTime:          m.endTime,
      words:            seg?.words,
      confidence:       parseFloat(confidence.toFixed(3)),
      textSimilarity:   parseFloat(m.similarity.toFixed(3)),
      temporalScore:    parseFloat(temporal.toFixed(3)),
      neighborScore:    parseFloat(neighbor.toFixed(3)),
      transcriptIndex:  m.transcriptIndex,
      isChorus:         m.isChorus,
      chorusGroupId:    chorus?.id,
      chorusOccurrence: m.isChorus ? chorusOccurrenceForTime(
        chorus!,
        transcript.at(-1)?.endTime ?? 0,
        official.length,
        m.transcriptIndex,
        transcript.length,
      ) : undefined,
      drift:            0,
      anchorDistance:   nearestAnchorDist === Infinity ? -1 : nearestAnchorDist,
    };
  });
}

// ── Stats ──────────────────────────────────────────────────────────────────────

function computeStats(lines: AlignedLine[], anchors: DriftAnchor[]): MatchStats {
  const drifts    = lines.map((l) => l.drift);
  const flagged   = lines.map((l, i) => ({ i, l })).filter(({ l }) => l.confidence < 0.45).map(({ i }) => i);
  const unmatched = lines.map((l, i) => ({ i, l })).filter(({ l }) => l.transcriptIndex < 0).map(({ i }) => i);
  const choruses  = new Set(lines.filter((l) => l.isChorus).map((l) => l.chorusGroupId)).size;

  return {
    avgConfidence: parseFloat((lines.reduce((s, l) => s + l.confidence, 0) / Math.max(1, lines.length)).toFixed(3)),
    avgSimilarity: parseFloat((lines.reduce((s, l) => s + l.textSimilarity, 0) / Math.max(1, lines.length)).toFixed(3)),
    chorusGroups:  choruses,
    totalAnchors:  anchors.length,
    maxDrift:      drifts.length ? Math.max(...drifts) : 0,
    minDrift:      drifts.length ? Math.min(...drifts) : 0,
    flaggedLines:  flagged,
    unmatched,
  };
}

// ── Public API ─────────────────────────────────────────────────────────────────

/**
 * Match official lyrics to transcript segments.
 *
 * @param transcript   WhisperX aligned segments
 * @param official     Provider lyrics (correct text, no times)
 * @param options      Tuning parameters
 */
export function matchLyricLines(
  transcript: TranscriptSegment[],
  official:   OfficialLine[],
  options:    MatchOptions = {},
): MatchResult {
  const opts = opt(options);

  if (!transcript.length || !official.length) {
    return { lines: [], stats: computeStats([], []), anchors: [] };
  }

  // ── 1. Normalize ─────────────────────────────────────────────────────────────
  const { norm: normOff,  romaji: romajiOff  } = normalizeAll(official.map((l) => l.text));
  const { norm: normTx,   romaji: romajiTx   } = normalizeAll(transcript.map((s) => s.text));

  // ── 2. Chorus detection ───────────────────────────────────────────────────────
  const chorusGroups = detectChoruses(normOff, opts.chorusThreshold, opts.minChorusLines);
  const chorusMap    = new Map<number, ChorusGroup>();
  for (const g of chorusGroups) {
    for (const idx of g.lineIndices) chorusMap.set(idx, g);
  }

  // ── 3. Similarity matrix ──────────────────────────────────────────────────────
  const simMatrix = buildSimilarityMatrix(normOff, normTx, romajiOff, romajiTx, opts);
  const costMatrix = simMatrix.map((row) => row.map((s) => 1 - s));

  // ── 4. DTW alignment ──────────────────────────────────────────────────────────
  const window = dtwWindow(opts, transcript.length);
  let mapping: number[];

  if (official.length > LONG_SONG_THRESHOLD) {
    mapping = segmentedDTW(costMatrix, 30, 8, window);
  } else {
    const { path } = dtw(costMatrix, window);
    mapping = resolvePathToMapping(path, official.length, costMatrix);
  }

  // ── 5. Build raw matches ──────────────────────────────────────────────────────
  let raw = buildRawMatches(mapping, official, transcript, simMatrix, chorusMap);

  // ── 6. Chorus disambiguation ──────────────────────────────────────────────────
  raw = disambiguateChoruses(raw, chorusGroups, transcript);

  // ── 7. Future-window re-scoring ───────────────────────────────────────────────
  raw = futureWindowRescore(raw, transcript, simMatrix, opts);

  // ── 8. Drift correction ───────────────────────────────────────────────────────
  const { corrected, anchors } = applyDriftCorrection(raw, opts);

  // ── 9. Final AlignedLine list ─────────────────────────────────────────────────
  const lines = buildAlignedLines(corrected, official, transcript, chorusMap, anchors);

  return { lines, stats: computeStats(lines, anchors), anchors };
}
