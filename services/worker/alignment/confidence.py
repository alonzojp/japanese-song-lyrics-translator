"""
Alignment confidence scoring.

Produces a per-run confidence report:
  - word_coverage:   fraction of words that have real (non-synthetic) timing
  - avg_word_score:  mean WhisperX/faster-whisper word confidence
  - gap_anomalies:   segments where the gap before them is unusually long
  - flagged_lines:   line indices (0-based) with low alignment confidence
  - overall:         composite 0.0–1.0 score
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


LOW_SCORE_THRESHOLD  = 0.40   # words below this are "uncertain"
GAP_MULTIPLIER       = 3.0    # gap > N × median_gap flags as anomaly
MIN_LINE_CONFIDENCE  = 0.35   # lines below this are flagged


@dataclass
class WordStats:
    total:        int = 0
    with_timing:  int = 0    # non-synthetic (score > 0)
    low_score:    int = 0
    score_sum:    float = 0.0


@dataclass
class AlignmentConfidence:
    overall:          float
    word_coverage:    float
    avg_word_score:   float
    low_score_words:  int
    gap_anomalies:    list[int]      # line indices
    flagged_lines:    list[int]      # line indices (low confidence)
    method:           str
    line_confidences: list[float]    # per-line score
    notes:            list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall":         round(self.overall, 3),
            "wordCoverage":    round(self.word_coverage, 3),
            "avgWordScore":    round(self.avg_word_score, 3),
            "lowScoreWords":   self.low_score_words,
            "gapAnomalies":    self.gap_anomalies,
            "flaggedLines":    self.flagged_lines,
            "method":          self.method,
            "lineConfidences": [round(s, 3) for s in self.line_confidences],
            "notes":           self.notes,
        }


def compute_confidence(
    segments: list[dict],
    method:   str,
) -> AlignmentConfidence:
    """
    Compute alignment confidence from enriched segments.
    segments: list of {start, end, text, words: [{word, start, end, score}]}
    """
    stats        = WordStats()
    line_scores:  list[float] = []
    line_gaps:    list[float] = []
    prev_end      = 0.0
    notes:        list[str]   = []

    for seg in segments:
        words      = seg.get("words") or []
        seg_start  = seg.get("start", 0.0)
        gap        = max(0.0, seg_start - prev_end)
        line_gaps.append(gap)
        prev_end   = seg.get("end", seg_start)

        w_scores: list[float] = []
        for w in words:
            stats.total      += 1
            score             = w.get("score", 0.0)
            stats.score_sum  += score
            if score > 0.0:
                stats.with_timing += 1
            if score < LOW_SCORE_THRESHOLD:
                stats.low_score += 1
            w_scores.append(score)

        line_conf = sum(w_scores) / len(w_scores) if w_scores else 0.0
        line_scores.append(line_conf)

    # Word coverage
    word_coverage  = stats.with_timing / stats.total if stats.total > 0 else 0.0
    avg_word_score = stats.score_sum  / stats.total if stats.total > 0 else 0.0

    # Gap anomalies
    if line_gaps:
        sorted_gaps = sorted(g for g in line_gaps if g > 0)
        median_gap  = sorted_gaps[len(sorted_gaps) // 2] if sorted_gaps else 0.0
        threshold   = median_gap * GAP_MULTIPLIER if median_gap > 0 else float("inf")
        gap_anomalies = [i for i, g in enumerate(line_gaps) if g > threshold and g > 2.0]
    else:
        gap_anomalies = []

    # Flagged lines
    flagged = [i for i, s in enumerate(line_scores) if s < MIN_LINE_CONFIDENCE]

    # Method penalty
    method_factor = {
        "whisperx_official": 1.00,   # acoustic + correct lyrics text = gold standard
        "whisperx_forced":   1.00,   # forced CTC alignment against known lyrics
        "whisperx":          1.00,
        "faster-whisper":    0.95,
        "openai-whisper":    0.85,
        "transcriber":       0.90,
        "aeneas":            0.70,
        "gentle":            0.55,
        "even":              0.20,
    }.get(method, 0.60)

    # Composite score
    coverage_score = word_coverage
    score_weight   = avg_word_score
    flag_penalty   = 1.0 - 0.5 * (len(flagged) / max(len(line_scores), 1))
    overall = method_factor * coverage_score * (0.4 + 0.6 * score_weight) * flag_penalty

    # Notes
    if method == "even":
        notes.append("Word timestamps are synthetic (no forced aligner available)")
    if word_coverage < 0.5:
        notes.append(f"Low word timestamp coverage: {word_coverage:.0%}")
    if len(flagged) > len(line_scores) * 0.3:
        notes.append(f"{len(flagged)}/{len(line_scores)} lines have low confidence")
    if gap_anomalies:
        notes.append(f"{len(gap_anomalies)} unusually long gaps detected")

    return AlignmentConfidence(
        overall=round(max(0.0, min(1.0, overall)), 3),
        word_coverage=round(word_coverage, 3),
        avg_word_score=round(avg_word_score, 3),
        low_score_words=stats.low_score,
        gap_anomalies=gap_anomalies,
        flagged_lines=flagged,
        method=method,
        line_confidences=line_scores,
        notes=notes,
    )
