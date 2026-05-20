"""
Step 4 — Post-processing: build LyricLine objects, compute confidence,
write all intermediate and final output files.

Inputs:  aligned_words.json (from step 3)
Outputs:
  aligned_lines.json   → LyricLine[]    (primary karaoke output)
  alignment_meta.json  → confidence report
  lyrics.json          → legacy backwards-compat format

Cache behaviour: if align stage is complete and aligned_lines.json exists,
loads and returns without reprocessing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import soundfile as sf

from alignment import (
    build_lyric_lines,
    compute_confidence,
    load_aligned_lines,
    save_aligned_lines,
    save_alignment_meta,
)
from cache import is_stage_complete, mark_stage_complete
from config import WHISPER_MODEL

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[int], None]]


def run_alignment_postprocess(
    aligned_segments: list[dict],
    output_dir:       Path,
    youtube_id:       str,
    audio_path:       Path,
    backend:          str,
    method:           str,
    vocals_only:      bool,
    progress_cb:      ProgressCallback = None,
    job_log=None,
) -> tuple[list[dict], dict]:
    """
    Build final LyricLine list and write all output files.
    Returns (lyric_lines, confidence_dict).
    """
    def log(msg: str) -> None:
        if job_log:
            job_log.info(msg, stage="align")
        logger.info(msg)

    # ── Cache hit ──────────────────────────────────────────────────────────────
    if is_stage_complete(youtube_id, "align"):
        cached = load_aligned_lines(output_dir)
        if cached:
            log(f"Cache hit — loaded {len(cached)} lines for {youtube_id}")
            if progress_cb:
                progress_cb(100)
            return cached, {}

    log(f"Post-processing {len(aligned_segments)} segments → LyricLine …")
    if progress_cb:
        progress_cb(10)

    # ── Confidence scoring ─────────────────────────────────────────────────────
    conf = compute_confidence(aligned_segments, method)
    log(
        f"Confidence: overall={conf.overall:.2f} "
        f"coverage={conf.word_coverage:.0%} "
        f"avg_score={conf.avg_word_score:.2f} "
        f"flagged={len(conf.flagged_lines)}"
    )
    if conf.notes:
        for note in conf.notes:
            if job_log:
                job_log.warning(note, stage="align")

    if progress_cb:
        progress_cb(30)

    # ── Build LyricLine list ───────────────────────────────────────────────────
    lines = build_lyric_lines(aligned_segments, line_confidences=conf.line_confidences)
    log(f"Built {len(lines)} LyricLine objects")

    if progress_cb:
        progress_cb(60)

    # ── Audio duration ─────────────────────────────────────────────────────────
    try:
        duration = sf.info(str(audio_path)).duration
    except Exception:
        duration = 0.0

    # ── Write output files ─────────────────────────────────────────────────────
    save_aligned_lines(output_dir, lines)
    save_alignment_meta(
        output_dir,
        youtube_id=youtube_id,
        backend=backend,
        method=method,
        confidence=conf.to_dict(),
        audio_duration=duration,
        whisper_model=WHISPER_MODEL,
        vocals_only=vocals_only,
    )

    log(f"Wrote aligned_lines.json ({len(lines)} lines), alignment_meta.json")
    if progress_cb:
        progress_cb(90)

    # ── Update manifest ────────────────────────────────────────────────────────
    mark_stage_complete(youtube_id, "align", {
        "lyricsPath":        "lyrics.json",
        "alignedLinesPath":  "aligned_lines.json",
        "alignmentMetaPath": "alignment_meta.json",
        "lineCount":         len(lines),
        "confidence":        conf.overall,
        "flaggedLines":      conf.flagged_lines,
    })

    if progress_cb:
        progress_cb(100)
    return lines, conf.to_dict()
