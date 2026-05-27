"""
Catastrophic ASR failure detector.

Fires before run_alignment_postprocess() and returns whether the WhisperX
output is so degraded that forced acoustic alignment should be used instead.

Failure conditions (any one triggers):
  segment_density   < 0.35   whisper collapsed many lyric lines into a few segments
  word_coverage     < 0.25   fewer than 1-in-4 words have real CTC timestamps
  overall_confidence< 0.35   composite quality metric bottomed out
  flagged_line_ratio> 0.40   > 40 % of segments have low word-score
  duplicate_start_ratio> 0.30 many segments share the same start time (chorus collapse)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from alignment.confidence import compute_confidence


@dataclass
class GateResult:
    triggered: bool
    reasons:   list[str]
    metrics:   dict[str, float]

    def summary(self) -> str:
        if not self.triggered:
            return f"ASR quality OK — metrics={self.metrics}"
        return f"CATASTROPHIC FAILURE — {self.reasons} — metrics={self.metrics}"


def detect_catastrophic_failure(
    segments:        list[dict],
    official_lines:  list[dict],   # [{text, ...}] from lyrics provider
    align_method:    str,
    job_log=None,
) -> GateResult:
    """
    Examine the raw aligned_segments (WhisperX output) and decide whether
    acoustic forced alignment should replace the existing result.

    Parameters
    ----------
    segments       : enriched segments from force_align() in aligner.py
    official_lines : official lyric lines from the lyrics provider
    align_method   : the method string used in force_align()
    job_log        : optional JobLogger for detailed diagnostic output
    """
    reasons: list[str] = []
    metrics: dict[str, float] = {}

    n_segs  = len(segments)
    n_lines = len([l for l in official_lines if l.get("text", "").strip()])

    if n_lines == 0:
        # No official lyrics → gate is meaningless, do not trigger
        return GateResult(False, [], {})

    # ── 1. Segment density ─────────────────────────────────────────────────────
    seg_density = n_segs / n_lines
    metrics["segment_density"]  = round(seg_density, 3)
    metrics["whisper_segments"] = n_segs
    metrics["lyric_lines"]      = n_lines
    if seg_density < 0.35:
        reasons.append(
            f"segment_density={seg_density:.3f} < 0.35 "
            f"(whisper={n_segs} segments for {n_lines} lyric lines)"
        )

    # ── 2. Word coverage + confidence ──────────────────────────────────────────
    conf = compute_confidence(segments, align_method)
    metrics["word_coverage"]      = conf.word_coverage
    metrics["overall_confidence"] = conf.overall
    metrics["flagged_lines"]      = len(conf.flagged_lines)

    flagged_ratio = len(conf.flagged_lines) / max(n_segs, 1)
    metrics["flagged_line_ratio"] = round(flagged_ratio, 3)

    if conf.word_coverage < 0.25:
        reasons.append(
            f"word_coverage={conf.word_coverage:.3f} < 0.25"
        )
    if conf.overall < 0.35:
        reasons.append(
            f"overall_confidence={conf.overall:.3f} < 0.35"
        )
    if flagged_ratio > 0.40:
        reasons.append(
            f"flagged_line_ratio={flagged_ratio:.3f} > 0.40 "
            f"({len(conf.flagged_lines)}/{n_segs} segments low-confidence)"
        )

    # ── 3. Chorus collapse: many segments share the same start time ────────────
    starts_rounded = [round(seg.get("start", 0.0), 1) for seg in segments]
    if starts_rounded:
        dup_count = sum(
            1 for _t, cnt in Counter(starts_rounded).items() if cnt > 1
        )
        dup_ratio = dup_count / len(starts_rounded)
        metrics["duplicate_start_ratio"] = round(dup_ratio, 3)
        if dup_ratio > 0.30:
            reasons.append(
                f"duplicate_start_ratio={dup_ratio:.3f} > 0.30 "
                f"(likely repeated-chorus collapse)"
            )

    triggered = len(reasons) > 0

    if job_log:
        if triggered:
            job_log.info(
                f"[quality_gate] TRIGGERED — catastrophic ASR failure: {reasons}",
                stage="align",
            )
            job_log.info(
                f"[quality_gate] Metrics: {metrics}",
                stage="align",
            )
        else:
            job_log.info(
                f"[quality_gate] ASR quality OK: metrics={metrics}",
                stage="align",
            )

    return GateResult(triggered=triggered, reasons=reasons, metrics=metrics)
