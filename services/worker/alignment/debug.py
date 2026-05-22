"""
Alignment debug instrumentation — observability only, no behavior changes.

Collects timing signals, split metrics, boundary evidence, and drift analysis
for every segment and line produced by _align_whisperx_with_official_lyrics().
Writes alignment_debug.json to the cache directory after each run.

Usage: instantiate AlignmentDebugger, call its record_* methods alongside
existing alignment logic, call save() at the end.  Nothing in this module
affects timestamps or fallback decisions.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AlignmentDebugger:
    """
    Accumulates per-segment and per-line alignment signals during
    _align_whisperx_with_official_lyrics() and writes a single JSON report.
    """

    def __init__(self, n_rough_segs: int, n_lines: int) -> None:
        self.n_rough_segs  = n_rough_segs
        self.n_lines       = n_lines
        self.segments:  list[dict] = []
        self.lines:     list[dict] = []
        self._cumulative_drift_ms = 0.0

    # ── Segment-level recording ───────────────────────────────────────────────

    def record_segment(
        self,
        seg_id:        int,
        vad_start:     float,
        vad_end:       float,
        matched_segs:  list[dict],
        line_texts:    list[str],
        chars:         list[dict],
        words:         list[dict],
    ) -> None:
        """
        Record baseline timing signals for one rough segment and the
        word/char data recovered by reconciliation.
        """
        duration = round(vad_end - vad_start, 3)

        # CTC signals: first/last word timestamps from reconciled sub-segments
        first_word_start = words[0].get('start') if words else None
        last_word_end    = words[-1].get('end')   if words else None
        first_char_start = chars[0].get('start')  if chars else None

        # CTC lag: gap between VAD boundary and first CTC word timestamp
        ctc_lag_ms: Optional[float] = None
        if first_word_start is not None:
            ctc_lag_ms = round((first_word_start - vad_start) * 1000, 1)

        self.segments.append({
            'segment_id':      seg_id,
            'vad_start':       vad_start,
            'vad_end':         vad_end,
            'duration':        duration,
            'n_lines_assigned': len(line_texts),
            'n_matched_subsegs': len(matched_segs),
            'first_word_start': first_word_start,
            'first_char_start': first_char_start,
            'last_word_end':    last_word_end,
            'word_count':       len(words),
            'char_count':       len(chars),
            'ctc_lag_ms':       ctc_lag_ms,
            'line_texts':       [t[:40] for t in line_texts],
        })

    # ── Line-level recording ──────────────────────────────────────────────────

    def record_line(
        self,
        line_idx:       int,
        line:           dict,
        seg_id:         int,
        vad_start:      float,
        vad_end:        float,
        char_ratio:     float,
        word_ratio:     float,
        timing_source:  str,
        words:          list[dict],
        silence_info:   Optional[dict] = None,
    ) -> None:
        """
        Record per-line split metrics, boundary evidence, and drift vs
        what a naive proportional model would have predicted.
        """
        start   = line.get('start', 0.0)
        end     = line.get('end',   0.0)
        dur     = round(end - start, 3)
        seg_dur = max(0.001, vad_end - vad_start)

        # Expected start: where a perfectly proportional model would place it
        # (assumes this line's share of time == its char_ratio of segment duration)
        expected_start = round(vad_start + char_ratio * seg_dur, 3)
        drift_ms       = round((start - expected_start) * 1000, 1)
        self._cumulative_drift_ms += drift_ms
        cumulative_drift_ms = round(self._cumulative_drift_ms, 1)

        # Word gap at boundary: gap between last word of previous line and
        # first word of this line (only meaningful when words have timestamps)
        word_gap_ms: Optional[float] = None
        if words and len(words) > 0:
            first_w = words[0].get('start')
            if first_w is not None:
                word_gap_ms = round((first_w - start) * 1000, 1)

        self.lines.append({
            'line_index':         line_idx,
            'segment_id':         seg_id,
            'line_text':          line.get('text', '')[:50],
            'timing_source':      timing_source,
            'start_time':         start,
            'end_time':           end,
            'duration_s':         dur,
            'char_ratio_used':    round(char_ratio, 4),
            'word_ratio_used':    round(word_ratio, 4),
            'expected_start':     expected_start,
            'drift_ms':           drift_ms,
            'cumulative_drift_ms': cumulative_drift_ms,
            'word_gap_at_boundary_ms': word_gap_ms,
            'silence_detected_near_boundary': silence_info.get('detected', False) if silence_info else False,
            'pause_length_ms':    silence_info.get('pause_ms')       if silence_info else None,
            'energy_drop_score':  silence_info.get('energy_score')   if silence_info else None,
            'has_word_timestamps': bool(words),
        })

    # ── Silence boundary probe ────────────────────────────────────────────────

    def probe_silence(
        self,
        audio_path: Path,
        probe_time: float,
        window_ms:  float = 300.0,
    ) -> dict:
        """
        Non-destructively probe for a silence/pause near a line boundary.
        Returns a dict with detected, pause_ms, energy_score.
        Does not affect any alignment decisions.
        """
        try:
            import soundfile as sf
            import numpy as np

            info = sf.info(str(audio_path))
            sr   = info.samplerate
            s    = max(0.0, probe_time - window_ms / 1000)
            e    = min(info.duration, probe_time + window_ms / 1000)

            s_fr  = int(s * sr)
            n_fr  = max(1, int((e - s) * sr))
            audio, _ = sf.read(str(audio_path), start=s_fr, frames=n_fr, dtype='float32')
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            hop = max(1, int(10 * sr / 1000))   # 10ms windows
            win = hop * 2
            n   = max(1, (len(audio) - win) // hop)
            rms = [float(np.sqrt(np.mean(audio[i*hop:i*hop+win]**2) + 1e-10))
                   for i in range(n)]

            peak    = max(rms) + 1e-10
            min_rms = min(rms)
            energy_score = round(1.0 - min_rms / peak, 3)   # 1.0 = total silence at some point

            # Find the quietest window and its timestamp
            min_idx = rms.index(min_rms)
            min_t   = s + min_idx * hop / sr
            pause_ms = round((probe_time - min_t) * 1000, 1)  # offset from boundary

            detected = energy_score > 0.5   # meaningful energy drop detected

            return {'detected': detected, 'pause_ms': pause_ms, 'energy_score': energy_score}

        except Exception:
            return {'detected': False, 'pause_ms': None, 'energy_score': None}

    # ── Summary and save ──────────────────────────────────────────────────────

    def save(self, output_dir: Path) -> None:
        """Compute summary metrics and write alignment_debug.json."""

        n = max(1, len(self.lines))

        # Source breakdown
        source_counts: dict[str, int] = {}
        for l in self.lines:
            src = l['timing_source']
            source_counts[src] = source_counts.get(src, 0) + 1

        def pct(k: str) -> float:
            return round(100 * source_counts.get(k, 0) / n, 1)

        # Drift analysis
        drifts       = [abs(l['drift_ms']) for l in self.lines]
        mean_drift   = round(sum(drifts) / n, 1) if drifts else 0.0
        worst_drift  = max(drifts) if drifts else 0.0
        worst_line   = next((l['line_index'] for l in self.lines
                             if abs(l['drift_ms']) == worst_drift), -1)

        # CTC lag analysis across segments
        ctc_lags = [s['ctc_lag_ms'] for s in self.segments if s['ctc_lag_ms'] is not None]
        mean_ctc_lag = round(sum(ctc_lags) / len(ctc_lags), 1) if ctc_lags else None

        # Lines with no silence evidence
        no_silence = sum(1 for l in self.lines if not l['silence_detected_near_boundary'])

        summary = {
            'total_lines':             n,
            'total_segments':          len(self.segments),
            'pct_vad_anchor':          pct('vad_anchor'),
            'pct_ctc_word':            pct('word_boundary'),
            'pct_char_split':          pct('char_boundary'),
            'pct_silence_split':       pct('silence_split'),
            'pct_proportional':        pct('proportional'),
            'pct_no_silence_evidence': round(100 * no_silence / n, 1),
            'mean_abs_drift_ms':       mean_drift,
            'worst_drift_ms':          round(worst_drift, 1),
            'worst_drift_line_idx':    worst_line,
            'mean_ctc_lag_ms':         mean_ctc_lag,
            'cumulative_drift_ms':     round(self._cumulative_drift_ms, 1),
            'source_counts':           source_counts,
            'generated_at':            datetime.now(timezone.utc).isoformat(),
        }

        payload = {
            'summary':  summary,
            'segments': self.segments,
            'lines':    self.lines,
        }

        out = output_dir / 'alignment_debug.json'
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        logger.info(
            "[debug] alignment_debug.json written: %d lines, mean_drift=%.0fms, "
            "ctc_lag=%.0fms, worst_line=%d (%.0fms)",
            n, mean_drift, mean_ctc_lag or 0, worst_line, worst_drift,
        )
