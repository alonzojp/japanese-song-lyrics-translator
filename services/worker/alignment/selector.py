"""
Alignment candidate selection.

Compares the acoustic alignment (aligned_lines.json + alignment_meta.json)
against the offline DTW match (matched_lyrics.json) and returns whichever
candidate has higher verifiable quality.

Selection factors:
  - Method trust level  (whisperx_official is gold standard)
  - DTW text similarity (how well the official lyrics matched the transcript)
  - DTW composite confidence
  - Timestamp validity  (monotonic, no zeros, no absurd gaps)

Always writes alignment_selection.json to the result dir for debugging.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Method trust levels ────────────────────────────────────────────────────────
# Acoustic methods that produced aligned_lines.json — higher = more trusted.
_METHOD_TRUST: dict[str, float] = {
    "whisperx_official": 1.00,   # acoustic + correct lyrics text = gold standard
    "whisperx_forced":   0.95,   # forced CTC alignment against known lyrics (bad-ASR rescue)
    "whisperx":          0.75,
    "faster-whisper":    0.70,
    "transcriber":       0.65,
    "cached":            0.60,
    "aeneas":            0.55,
    "gentle":            0.45,
    "even":              0.10,
    "unknown":           0.50,
}

# Minimum DTW avgSimilarity to override acoustic, keyed by acoustic method.
# Higher-quality acoustic methods require a stronger DTW match before being displaced.
_MIN_DTW_SIM: dict[str, float] = {
    "whisperx_official": 0.65,
    "whisperx_forced":   0.60,   # forced alignment on bad ASR song — need strong DTW to displace
    "whisperx":          0.50,
    "faster-whisper":    0.50,
    "transcriber":       0.50,
    "cached":            0.45,
    "aeneas":            0.35,
    "gentle":            0.30,
    "even":              0.25,
    "unknown":           0.40,
}

# Minimum DTW avgConfidence to override acoustic.
_MIN_DTW_CONF: dict[str, float] = {
    "whisperx_official": 0.58,
    "whisperx_forced":   0.55,
    "whisperx":          0.48,
    "faster-whisper":    0.48,
    "transcriber":       0.48,
    "cached":            0.45,
    "aeneas":            0.40,
    "gentle":            0.35,
    "even":              0.35,
    "unknown":           0.40,
}


# ── Timestamp validation ───────────────────────────────────────────────────────

def _check_timestamps(lines: list[dict]) -> dict:
    """
    Validate timestamp consistency of a LyricLine list.
    Returns issue counts: non_monotonic, negative_duration, zero_timestamps, giant_gaps.
    """
    issues = dict(non_monotonic=0, negative_duration=0, zero_timestamps=0, giant_gaps=0)
    prev_end = -1.0
    for line in lines:
        start = line.get("startTime", 0.0)
        end   = line.get("endTime",   0.0)
        if start < prev_end - 0.1:
            issues["non_monotonic"] += 1
        if end < start:
            issues["negative_duration"] += 1
        if start == 0.0 and end == 0.0:
            issues["zero_timestamps"] += 1
        if start - prev_end > 30.0 and prev_end >= 0:
            issues["giant_gaps"] += 1
        prev_end = max(prev_end, end)
    return issues


def _ts_validity(issues: dict, n: int) -> float:
    """Convert issue counts to a 0.0–1.0 validity multiplier."""
    if n == 0:
        return 0.0
    penalty = (
        issues["non_monotonic"]     * 0.15 +
        issues["negative_duration"] * 0.20 +
        issues["zero_timestamps"]   * 0.10 +
        issues["giant_gaps"]        * 0.05
    ) / n
    return max(0.0, 1.0 - min(1.0, penalty * 5))


# ── Timestamp repair ──────────────────────────────────────────────────────────

def _repair_timestamps(lines: list[dict]) -> list[dict]:
    """
    Clamp minor non-monotonic timestamps to restore strict ordering.
    Creates shallow copies — does not mutate the input.
    """
    repaired = [dict(line) for line in lines]
    prev_end = -1.0
    for line in repaired:
        start = line.get("startTime", 0.0)
        end   = line.get("endTime",   0.0)
        if start < prev_end - 0.1:
            line["startTime"] = round(prev_end + 0.05, 3)
            if end < line["startTime"]:
                line["endTime"] = round(line["startTime"] + 0.5, 3)
        prev_end = max(prev_end, line.get("endTime", 0.0))
    return repaired


# ── Per-candidate scoring ──────────────────────────────────────────────────────

# Gap thresholds for playback quality assessment
_UNEXPLAINED_GAP_MIN_S = 0.75   # gaps > this (but < instrumental) are "unexplained"
_INSTRUMENTAL_GAP_S    = 20.0   # gaps > this are assumed intentional (bridge/outro)
_SHORT_LINE_S          = 1.5    # lines shorter than this hurt readability
_LONG_LINE_S           = 8.0    # lines longer than this hurt readability


def _score_playback(lines: list[dict], job_log=None) -> float:
    """
    Playback continuity and readability score [0.0–1.0].

    Penalises:
      - Repeated unexplained gaps (0.75s–20s) between lines
      - Excessive short lines (< 1.5s)
      - Excessive long lines (> 8s)

    Does NOT penalise large intentional gaps (> 20s = instrumental/bridge).
    """
    n = len(lines)
    if n < 2:
        return 1.0

    gaps  = []
    durs  = []
    for i, line in enumerate(lines):
        start = line.get("startTime", 0.0)
        end   = line.get("endTime",   0.0)
        durs.append(max(0.0, end - start))
        if i + 1 < n:
            gap = lines[i + 1].get("startTime", end) - end
            gaps.append(gap)

    avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
    max_gap = max(gaps) if gaps else 0.0
    unexplained = [g for g in gaps if _UNEXPLAINED_GAP_MIN_S < g < _INSTRUMENTAL_GAP_S]
    short_lines = sum(1 for d in durs if d < _SHORT_LINE_S)
    long_lines  = sum(1 for d in durs if d > _LONG_LINE_S)

    # Continuous pairs (gap ≤ 0.5s)
    contiguous_pairs = sum(1 for g in gaps if g <= 0.50)

    # Penalties: each unexplained gap = 4%, each short = 2%, each long = 1.5%
    gap_penalty   = min(0.60, len(unexplained) / n * 0.90)
    short_penalty = min(0.20, short_lines / n * 0.40)
    long_penalty  = min(0.15, long_lines  / n * 0.30)

    score = max(0.0, 1.0 - gap_penalty - short_penalty - long_penalty)

    if job_log:
        job_log.info(
            f"[playback_quality] "
            f"avg_gap={avg_gap:.2f}s max_gap={max_gap:.1f}s "
            f"unexplained_gaps={len(unexplained)}/{len(gaps)} "
            f"short_lines={short_lines} long_lines={long_lines} "
            f"contiguous_pairs={contiguous_pairs}/{len(gaps)} "
            f"continuity={score:.3f}",
            stage="align",
        )

    return round(score, 4)


def _score_acoustic(method: str, confidence_dict: dict, lines: list[dict]) -> float:
    """
    Composite quality score for the acoustic alignment.
    score = method_trust × overall_confidence × timestamp_validity
    """
    trust    = _METHOD_TRUST.get(method, 0.50)
    conf     = confidence_dict.get("overall", 0.50)
    ts_score = _ts_validity(_check_timestamps(lines), len(lines))
    return round(trust * conf * ts_score, 4)


def _score_dtw(stats: dict, lines: list[dict], job_log=None) -> float:
    """
    Composite quality score for the DTW-matched alignment.

    Weights:
      35% text similarity  (how well official lyrics matched the transcript)
      30% confidence       (DTW composite per-line confidence)
      10% anchor bonus     (high-confidence anchor density)
      15% flag penalty     (low-confidence lines)
      25% playback quality (gap continuity + readability — NEW)

    Total raw ≤ 1.05 before ts_score and playback weighting.
    """
    n        = max(1, len(lines))
    avg_sim  = stats.get("avgSimilarity", 0.0)
    avg_conf = stats.get("avgConfidence", 0.0)
    anchors  = stats.get("totalAnchors",  0)
    flagged  = len(stats.get("flaggedLines", []))

    anchor_bonus    = min(0.10, anchors / n * 0.20)
    flag_penalty    = flagged / n * 0.15
    ts_score        = _ts_validity(_check_timestamps(lines), n)
    playback_score  = _score_playback(lines, job_log=job_log)

    raw = (
        0.35 * avg_sim
        + 0.30 * avg_conf
        + anchor_bonus
        - flag_penalty
        + 0.25 * playback_score
    )
    return round(max(0.0, raw) * ts_score, 4)


# ── Drift detection ────────────────────────────────────────────────────────────

def _detect_drift(acoustic_lines: list[dict], dtw_lines: list[dict]) -> dict:
    """
    Compare per-line timestamps between acoustic and DTW alignments.
    Returns max drift, mean drift, and whether large drift was detected.
    Large drift alone is not a red flag — it can mean DTW is correcting acoustic errors.
    Only block on drift when similarity is also poor (see selection logic).
    """
    if not acoustic_lines or not dtw_lines or len(acoustic_lines) != len(dtw_lines):
        return {"maxDrift": None, "meanDrift": None, "largeDriftDetected": False}

    drifts = [
        abs(d.get("startTime", 0.0) - a.get("startTime", 0.0))
        for a, d in zip(acoustic_lines, dtw_lines)
    ]
    max_drift  = max(drifts)
    mean_drift = sum(drifts) / len(drifts)
    return {
        "maxDrift":           round(max_drift,  3),
        "meanDrift":          round(mean_drift, 3),
        "largeDriftDetected": max_drift > 10.0,
    }


# ── Public entry point ─────────────────────────────────────────────────────────

def select_best_alignment(
    result_dir: Path,
    job_log=None,
) -> dict:
    """
    Compare acoustic alignment vs DTW match and return the better one.

    Returns:
      {
        source:           "acoustic" | "dtw"
        lines:            list[dict]   — the winning LyricLine list
        stats:            dict         — DTW stats if DTW won, else {}
        acoustic_score:   float
        dtw_score:        float
        selection_meta:   dict         — full debug info
      }

    Side effect: writes alignment_selection.json to result_dir.
    """
    def log(msg: str) -> None:
        if job_log:
            job_log.info(msg, stage="align")
        logger.info(msg)

    acoustic_path = result_dir / "aligned_lines.json"
    dtw_path      = result_dir / "matched_lyrics.json"
    meta_path     = result_dir / "alignment_meta.json"

    # ── Load acoustic alignment ──────────────────────────────────────────────
    acoustic_lines: list[dict] = []
    acoustic_method = "unknown"
    acoustic_conf_dict: dict = {}

    if acoustic_path.exists():
        raw = json.loads(acoustic_path.read_text(encoding="utf-8"))
        acoustic_lines = raw.get("lines", raw) if isinstance(raw, dict) else raw

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        acoustic_method    = meta.get("alignmentMethod", "unknown")
        acoustic_conf_dict = meta.get("confidence", {})

    acoustic_score = _score_acoustic(acoustic_method, acoustic_conf_dict, acoustic_lines) if acoustic_lines else 0.0

    # ── Load DTW alignment ───────────────────────────────────────────────────
    dtw_lines: list[dict] = []
    dtw_stats: dict = {}

    if dtw_path.exists():
        dtw_data  = json.loads(dtw_path.read_text(encoding="utf-8"))
        dtw_lines = dtw_data.get("lines", [])
        dtw_stats = dtw_data.get("stats", {})

    dtw_score = _score_dtw(dtw_stats, dtw_lines, job_log=job_log) if dtw_lines else 0.0

    # ── Drift analysis (when both available) ────────────────────────────────
    drift_info = _detect_drift(acoustic_lines, dtw_lines)

    # ── Selection decision ───────────────────────────────────────────────────
    selected = "acoustic"
    reason   = ""

    if not acoustic_lines and dtw_lines:
        selected = "dtw"
        reason   = "No acoustic alignment found — using DTW"

    elif not dtw_lines:
        selected = "acoustic"
        reason   = f"No DTW alignment found — using acoustic ({acoustic_method})"

    else:
        min_sim  = _MIN_DTW_SIM.get(acoustic_method,  0.40)
        min_conf = _MIN_DTW_CONF.get(acoustic_method, 0.40)
        avg_sim  = dtw_stats.get("avgSimilarity", 0.0)
        avg_conf = dtw_stats.get("avgConfidence", 0.0)
        ts_issues = _check_timestamps(dtw_lines)
        n = len(dtw_lines)

        # Hard blocks — DTW cannot win regardless of score
        if avg_sim < min_sim:
            selected = "acoustic"
            reason   = (
                f"DTW blocked: similarity {avg_sim:.2f} < threshold {min_sim:.2f} "
                f"for {acoustic_method} — keeping acoustic (score={acoustic_score:.3f})"
            )
        elif avg_conf < min_conf:
            selected = "acoustic"
            reason   = (
                f"DTW blocked: confidence {avg_conf:.2f} < threshold {min_conf:.2f} "
                f"for {acoustic_method} — keeping acoustic (score={acoustic_score:.3f})"
            )
        elif drift_info["largeDriftDetected"] and avg_sim < 0.50:
            # Large drift + poor text similarity = DTW misassigning lines, not correcting them.
            # Large drift + high similarity = DTW is correcting acoustic timing errors (let score decide).
            selected = "acoustic"
            reason   = (
                f"DTW blocked: large drift ({drift_info['maxDrift']:.1f}s) with "
                f"poor similarity {avg_sim:.2f} — likely misassignment, keeping acoustic ({acoustic_method})"
            )
        elif ts_issues["zero_timestamps"] > n * 0.10:
            selected = "acoustic"
            reason   = (
                f"DTW blocked: {ts_issues['zero_timestamps']}/{n} lines have zero timestamps "
                f"— keeping acoustic ({acoustic_method})"
            )
        elif ts_issues["non_monotonic"] > 0:
            nm = ts_issues["non_monotonic"]
            if nm <= 3 and dtw_score > acoustic_score + 0.05:
                # Minor inversion — attempt repair; allow DTW if score is clearly better
                dtw_lines = _repair_timestamps(dtw_lines)
                ts_after  = _check_timestamps(dtw_lines)
                if ts_after["non_monotonic"] == 0:
                    selected = "dtw"
                    reason   = (
                        f"DTW selected (repaired {nm} non-monotonic timestamp(s)): "
                        f"score {dtw_score:.3f} > acoustic {acoustic_score:.3f} "
                        f"(sim={avg_sim:.2f}, conf={avg_conf:.2f})"
                    )
                else:
                    selected = "acoustic"
                    reason   = (
                        f"DTW blocked: {ts_after['non_monotonic']} non-monotonic timestamp(s) "
                        f"remain after repair — keeping acoustic ({acoustic_method})"
                    )
            else:
                why = (
                    f"too many ({nm} > 3)"
                    if nm > 3 else
                    f"score gain {dtw_score - acoustic_score:.3f} < 0.05"
                )
                selected = "acoustic"
                reason   = (
                    f"DTW blocked: {nm} non-monotonic timestamp(s) [{why}] "
                    f"— keeping acoustic ({acoustic_method})"
                )
        # Score comparison — DTW wins only if measurably better
        elif dtw_score > acoustic_score:
            selected = "dtw"
            reason   = (
                f"DTW selected: score {dtw_score:.3f} > acoustic {acoustic_score:.3f} "
                f"(sim={avg_sim:.2f}, conf={avg_conf:.2f}, "
                f"anchors={dtw_stats.get('totalAnchors', 0)})"
            )
        else:
            selected = "acoustic"
            reason   = (
                f"Keeping acoustic ({acoustic_method}): score {acoustic_score:.3f} ≥ "
                f"DTW {dtw_score:.3f} (sim={avg_sim:.2f}, conf={avg_conf:.2f})"
            )

    log(f"[selector] {reason}")

    winning_lines = dtw_lines if selected == "dtw" else acoustic_lines

    acoustic_playback = _score_playback(acoustic_lines) if acoustic_lines else 0.0
    dtw_playback      = _score_playback(dtw_lines)      if dtw_lines      else 0.0

    selection_meta = {
        "selectedSource":        selected,
        "reason":                reason,
        "acousticMethod":        acoustic_method,
        "acousticScore":         acoustic_score,
        "acousticOverallConf":   acoustic_conf_dict.get("overall", 0.0),
        "acousticPlayback":      acoustic_playback,
        "dtwScore":              dtw_score,
        "dtwSimilarity":         round(dtw_stats.get("avgSimilarity", 0.0), 4),
        "dtwConfidence":         round(dtw_stats.get("avgConfidence", 0.0), 4),
        "dtwAnchors":            dtw_stats.get("totalAnchors", 0),
        "dtwFlaggedCount":       len(dtw_stats.get("flaggedLines", [])),
        "dtwPlayback":           dtw_playback,
        "drift":                 drift_info,
        "dtwTimestampIssues":    _check_timestamps(dtw_lines) if dtw_lines else {},
        "selectedAt":            datetime.now(timezone.utc).isoformat(),
    }

    (result_dir / "alignment_selection.json").write_text(
        json.dumps(selection_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "source":         selected,
        "lines":          winning_lines,
        "stats":          dtw_stats if selected == "dtw" else {},
        "acoustic_score": acoustic_score,
        "dtw_score":      dtw_score,
        "selection_meta": selection_meta,
    }
