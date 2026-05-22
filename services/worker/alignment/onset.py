"""
Onset timing instrumentation.

Measures the relationship between displayed lyric timestamps and actual
vocal onset in the audio. Writes onset_analysis.json for debugging.

WHY TIMESTAMPS LAG (diagnostics only — no corrections applied here):

  WhisperX CTC models mark the *stable recognition window* for each phoneme,
  not the acoustic onset. For sung Japanese this creates a systematic lag:

    Vocal onset  →  [100–300ms]  →  CTC recognition  →  timestamp

  For segment-boundary lines (first line of each Whisper segment) the lag is
  close to zero because Whisper's own VAD already detects speech onset.

  For within-segment lines (from proportional/char distribution) the lag
  depends on how accurately we split the segment window.

  In continuous singing (no silence between lines), onset detection via simple
  energy threshold finds the PREVIOUS line's tail, not the new onset. For
  these lines the measured lag saturates at the search window size and is
  not meaningful. This module detects and flags that condition.

Writes:
  onset_analysis.json  — per-line timing breakdown + summary stats
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SEARCH_BEFORE_MS = 700   # look this far before timestamp for onset
SEARCH_AFTER_MS  = 100   # look this far after timestamp
RMS_WIN_MS       = 10
RMS_HOP_MS       = 5
ONSET_HALF_PEAK  = 0.5   # threshold = this fraction of window peak


def _detect_segment_onset(
    audio_path: Path,
    timestamp:  float,
    sr:         int,
) -> Optional[float]:
    """
    Find the first moment in [timestamp - SEARCH_BEFORE_MS, timestamp + SEARCH_AFTER_MS]
    where RMS energy rises to 50% of the window's peak. Returns None if no clear onset.
    """
    try:
        import soundfile as sf
        import numpy as np

        s = max(0.0, timestamp - SEARCH_BEFORE_MS / 1000)
        e = timestamp + SEARCH_AFTER_MS / 1000
        s_fr = int(s * sr)
        n_fr = max(1, int((e - s) * sr))

        audio, _ = sf.read(str(audio_path), start=s_fr, frames=n_fr, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        hop = max(1, int(RMS_HOP_MS * sr / 1000))
        win = max(1, int(RMS_WIN_MS * sr / 1000))
        n   = max(1, (len(audio) - win) // hop)

        rms = [float((audio[i*hop:i*hop+win]**2).mean()**0.5 + 1e-10) for i in range(n)]
        pk  = max(rms)
        thr = pk * ONSET_HALF_PEAK

        for i, r in enumerate(rms):
            t = s + i * hop / sr
            if r >= thr and t < timestamp:
                return round(t, 3)

        return None
    except Exception:
        return None


def _is_segment_boundary(line_idx: int, lines: list[dict], segs: list[dict]) -> bool:
    """True if this line's startTime matches a Whisper segment start (within 100ms)."""
    ts = lines[line_idx].get('startTime', 0.0)
    return any(abs(ts - s.get('start', -999)) < 0.1 for s in segs)


def _gap_to_prev(line_idx: int, lines: list[dict]) -> float:
    """Gap in seconds between previous line end and this line start."""
    if line_idx == 0:
        return lines[0].get('startTime', 0.0)
    return lines[line_idx]['startTime'] - lines[line_idx - 1]['endTime']


def measure_and_log(
    lines:       list[dict],
    audio_path:  Path,
    output_dir:  Path,
    segs:        Optional[list[dict]] = None,
    job_log=None,
) -> None:
    """
    Measure onset timing characteristics for every lyric line and write
    onset_analysis.json.  No timestamps are modified.

    lines:      LyricLine list (startTime/endTime keys)
    audio_path: path to vocals_asr.wav (16 kHz mono)
    output_dir: cache dir for this video
    segs:       raw Whisper segments (for segment-boundary detection)
    """
    def log(msg: str) -> None:
        if job_log:
            job_log.info(msg, stage="align")
        logger.info(msg)

    try:
        import soundfile as sf
        info = sf.info(str(audio_path))
        sr   = info.samplerate
    except Exception:
        return

    segs = segs or []
    entries: list[dict] = []

    for i, line in enumerate(lines):
        ts       = line.get('startTime', 0.0)
        onset    = _detect_segment_onset(audio_path, ts, sr)
        raw_lag  = round((ts - onset) * 1000, 1) if onset is not None else None
        gap      = round(_gap_to_prev(i, lines), 3)
        is_bnd   = _is_segment_boundary(i, lines, segs)
        dur      = round(line.get('endTime', ts) - ts, 3)
        has_words = bool(line.get('words'))

        # Saturated detection: onset is almost exactly SEARCH_BEFORE_MS before
        # the timestamp — signals that previous vocal was still present (continuous singing).
        saturated = (
            raw_lag is not None and
            abs(raw_lag - SEARCH_BEFORE_MS) < 50  # within 50ms of full lookback
        )

        entries.append({
            'line':           i,
            'text':           line.get('text', '')[:35],
            'timestamp':      ts,
            'duration_s':     dur,
            'gap_to_prev_s':  gap,
            'onset':          onset,
            'raw_lag_ms':     raw_lag,
            'lag_saturated':  saturated,
            'is_seg_boundary': is_bnd,
            'has_words':      has_words,
            'source':         (
                'acoustic_word'  if has_words else
                'seg_boundary'   if is_bnd    else
                'proportional'
            ),
        })

    # ── Aggregate stats ──────────────────────────────────────────────────────
    boundary_lags  = [e['raw_lag_ms'] for e in entries if e['is_seg_boundary'] and e['raw_lag_ms'] is not None and not e['lag_saturated']]
    interior_lags  = [e['raw_lag_ms'] for e in entries if not e['is_seg_boundary'] and e['raw_lag_ms'] is not None and not e['lag_saturated']]
    saturated_cnt  = sum(1 for e in entries if e['lag_saturated'])
    no_words_cnt   = sum(1 for e in entries if not e['has_words'])

    def _avg(lst): return round(sum(lst)/len(lst), 1) if lst else None

    # Drift: does lag grow over time?
    measurable = [(e['timestamp'], e['raw_lag_ms']) for e in entries
                  if e['raw_lag_ms'] is not None and not e['lag_saturated']]
    drift_ms_per_s: Optional[float] = None
    if len(measurable) >= 4:
        ts_l = [x[0] for x in measurable]
        lg_l = [x[1] for x in measurable]
        n = len(ts_l); sx = sum(ts_l); sy = sum(lg_l)
        sxx = sum(t*t for t in ts_l); sxy = sum(t*l for t,l in zip(ts_l, lg_l))
        d = n*sxx - sx*sx
        if abs(d) > 1e-6:
            drift_ms_per_s = round((n*sxy - sx*sy)/d, 3)

    summary = {
        'total_lines':          len(lines),
        'no_word_timestamps':   no_words_cnt,
        'saturated_detections': saturated_cnt,
        'seg_boundary_lines':   sum(1 for e in entries if e['is_seg_boundary']),
        'avg_boundary_lag_ms':  _avg(boundary_lags),
        'avg_interior_lag_ms':  _avg(interior_lags),
        'drift_ms_per_sec':     drift_ms_per_s,
        'note': (
            'Saturated detections indicate continuous singing — previous vocal '
            'still present at the lookback boundary. These lags are not meaningful.'
        ),
    }

    log(
        f"[onset] {no_words_cnt}/{len(lines)} lines have no word-level timestamps. "
        f"Saturated detections (continuous singing): {saturated_cnt}. "
        f"Boundary lag avg: {summary['avg_boundary_lag_ms']}ms. "
        f"Drift: {drift_ms_per_s}ms/s."
    )
    if drift_ms_per_s and abs(drift_ms_per_s) > 0.5:
        log(
            f"[onset] WARNING: cumulative drift {drift_ms_per_s:+.2f}ms/s — "
            f"timestamps trend {'later' if drift_ms_per_s > 0 else 'earlier'} over the song"
        )

    (output_dir / 'onset_analysis.json').write_text(
        json.dumps({'summary': summary, 'lines': entries}, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
