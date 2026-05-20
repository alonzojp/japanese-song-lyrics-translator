/**
 * Dynamic Time Warping with Sakoe-Chiba band constraint.
 *
 * Aligns two sequences by finding the minimum-cost monotone path through
 * a cost matrix. The band constraint prevents extreme warping (which would
 * happen with repeated choruses if left unconstrained).
 *
 * Returns both the total distance and the optimal warp path.
 */

export interface DTWResult {
  distance: number;
  path:     Array<[number, number]>;   // [rowIndex, colIndex] pairs
}

const INF = Number.MAX_SAFE_INTEGER / 2;

/**
 * Classic DTW with optional Sakoe-Chiba band.
 *
 * @param cost   cost[i][j] = distance between row item i and col item j
 * @param window Sakoe-Chiba half-band width. Pass 0 for unconstrained.
 */
export function dtw(cost: number[][], window = 0): DTWResult {
  const n = cost.length;
  const m = cost[0]?.length ?? 0;
  if (n === 0 || m === 0) return { distance: 0, path: [] };

  // Effective window: must cover the diagonal even when lengths differ
  const effectiveWindow = window > 0
    ? Math.max(window, Math.abs(n - m))
    : Math.max(n, m);

  // Accumulated cost matrix
  const acc: number[][] = Array.from({ length: n }, () => new Array(m).fill(INF));
  acc[0]![0] = cost[0]![0]!;

  for (let i = 1; i < n; i++) {
    if (Math.abs(i - 0) <= effectiveWindow) acc[i]![0] = cost[i]![0]! + acc[i - 1]![0]!;
  }
  for (let j = 1; j < m; j++) {
    if (Math.abs(0 - j) <= effectiveWindow) acc[0]![j] = cost[0]![j]! + acc[0]![j - 1]!;
  }

  for (let i = 1; i < n; i++) {
    for (let j = 1; j < m; j++) {
      if (window > 0 && Math.abs(i - j) > effectiveWindow) continue;
      acc[i]![j] = cost[i]![j]! + Math.min(
        acc[i - 1]![j]!,        // insertion
        acc[i]![j - 1]!,        // deletion
        acc[i - 1]![j - 1]!     // match
      );
    }
  }

  return {
    distance: acc[n - 1]![m - 1]!,
    path:     traceback(acc, n, m),
  };
}

function traceback(acc: number[][], n: number, m: number): Array<[number, number]> {
  const path: Array<[number, number]> = [];
  let i = n - 1;
  let j = m - 1;
  path.push([i, j]);

  while (i > 0 || j > 0) {
    if (i === 0) {
      j--;
    } else if (j === 0) {
      i--;
    } else {
      const diag = acc[i - 1]![j - 1]!;
      const up   = acc[i - 1]![j]!;
      const left = acc[i]![j - 1]!;
      const best = Math.min(diag, up, left);
      if (best === diag) { i--; j--; }
      else if (best === up) { i--; }
      else { j--; }
    }
    path.push([i, j]);
  }

  return path.reverse();
}

/**
 * Given the DTW path, resolve each row index to its best-matching column.
 * When multiple columns map to the same row, prefer the one with lowest cost.
 *
 * Returns an array `mapping[i] = j` (official index → transcript index).
 */
export function resolvePathToMapping(
  path:  Array<[number, number]>,
  n:     number,   // number of official lines
  cost:  number[][]
): number[] {
  // Collect all (i, j) pairs from the path
  const candidates = new Map<number, number[]>();
  for (const [i, j] of path) {
    if (!candidates.has(i)) candidates.set(i, []);
    candidates.get(i)!.push(j);
  }

  const mapping = new Array<number>(n).fill(-1);
  for (let i = 0; i < n; i++) {
    const js = candidates.get(i);
    if (!js || js.length === 0) continue;
    // Pick j with lowest cost
    let best = js[0]!;
    for (const j of js) {
      if ((cost[i]![j] ?? 1) < (cost[i]![best] ?? 1)) best = j;
    }
    mapping[i] = best;
  }
  return mapping;
}

/**
 * Segment-wise DTW for long songs.
 *
 * Splits the official lyric sequence into overlapping windows and runs DTW
 * independently on each window, then stitches the results together.
 * This avoids quadratic memory for very long songs.
 *
 * @param cost     Full cost matrix [n × m]
 * @param segSize  Lines per segment (default 30)
 * @param overlap  Overlap between segments (default 8)
 * @param window   Sakoe-Chiba window per segment
 */
export function segmentedDTW(
  cost:    number[][],
  segSize  = 30,
  overlap  = 8,
  window   = 12,
): number[] {
  const n = cost.length;
  const m = cost[0]?.length ?? 0;
  const mapping = new Array<number>(n).fill(-1);

  let lastColEnd = 0;   // transcript boundary from last segment

  for (let iStart = 0; iStart < n; iStart += segSize - overlap) {
    const iEnd = Math.min(iStart + segSize, n);

    // Restrict transcript window based on where we are (monotone constraint)
    const expectedJ = Math.round((iStart / n) * m);
    const jStart    = Math.max(lastColEnd, expectedJ - window * 2);
    const jEnd      = Math.min(m, expectedJ + segSize + window * 2);

    if (jStart >= jEnd) continue;

    // Extract sub-cost matrix
    const subCost = cost.slice(iStart, iEnd).map((row) => row.slice(jStart, jEnd));

    const result  = dtw(subCost, window);
    const subMap  = resolvePathToMapping(result.path, iEnd - iStart, subCost);

    // Write only non-overlap rows (to avoid overwrite from next segment)
    const writeEnd = iStart === 0 ? iEnd : Math.min(iEnd, iStart + segSize - overlap);
    for (let i = iStart; i < writeEnd; i++) {
      const localI = i - iStart;
      if (subMap[localI] !== undefined && subMap[localI]! >= 0) {
        mapping[i] = jStart + subMap[localI]!;
      }
    }

    // Update transcript boundary for next segment
    const lastMapped = mapping.slice(iStart, writeEnd).filter(j => j >= 0);
    if (lastMapped.length > 0) {
      lastColEnd = Math.max(...lastMapped);
    }
  }

  return mapping;
}
