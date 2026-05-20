"""
Step 4 — Post-processing: convert WhisperX segments → clean TranscribedLine
objects, filter noise, and write lyrics.json.

Cache behaviour: skips if manifest marks align as complete and lyrics.json exists.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

from cache import is_stage_complete, mark_stage_complete
from models import LyricWord, TranscribedLine

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[int], None]]

MIN_SEGMENT_DURATION = 0.2  # seconds

# Characters that count as "Japanese" for filtering
_JA_RANGES = [
    (0x3040, 0x309F),  # hiragana
    (0x30A0, 0x30FF),  # katakana
    (0x4E00, 0x9FFF),  # CJK unified
    (0x3400, 0x4DBF),  # CJK extension A
]


def _has_japanese(text: str) -> bool:
    return any(
        lo <= ord(c) <= hi
        for c in text
        for lo, hi in _JA_RANGES
    )


def _line_to_dict(line: TranscribedLine) -> dict:
    return {
        "index":     line.index,
        "startTime": line.start_time,
        "endTime":   line.end_time,
        "text":      line.text,
        "words": [
            {"word": w.word, "startTime": w.start_time, "endTime": w.end_time, "score": w.score}
            for w in line.words
        ],
    }


def process_segments(
    raw_segments:  list[dict],
    output_dir:    Path,
    youtube_id:    str,
    audio_duration: float,
    vocals_only:   bool,
    whisper_model: str,
    progress_cb:   ProgressCallback = None,
    job_log=None,
) -> list[TranscribedLine]:
    """
    Convert WhisperX output to clean TranscribedLine list and write lyrics.json.
    """
    def log(msg: str) -> None:
        if job_log:
            job_log.info(msg, stage="align")
        logger.info(msg)

    lyrics_path = output_dir / "lyrics.json"

    # ── Cache hit ──────────────────────────────────────────────────────────────
    if is_stage_complete(youtube_id, "align") and lyrics_path.exists():
        log(f"Cache hit — loading saved lyrics for {youtube_id}")
        data = json.loads(lyrics_path.read_text(encoding="utf-8"))
        lines = [
            TranscribedLine(
                index=ln["index"],
                startTime=ln["startTime"],
                endTime=ln["endTime"],
                text=ln["text"],
                words=[
                    LyricWord(
                        word=w["word"],
                        startTime=w["startTime"],
                        endTime=w["endTime"],
                        score=w.get("score", 1.0),
                    )
                    for w in ln.get("words", [])
                ],
            )
            for ln in data.get("lyrics", [])
        ]
        if progress_cb:
            progress_cb(100)
        return lines

    lines: list[TranscribedLine] = []
    dropped = 0

    for seg in raw_segments:
        start = seg.get("start", 0.0)
        end   = seg.get("end",   start)
        text  = (seg.get("text") or "").strip()

        if end - start < MIN_SEGMENT_DURATION:
            dropped += 1
            continue
        if not _has_japanese(text):
            logger.debug("Dropping non-Japanese segment: %r", text)
            dropped += 1
            continue

        words: list[LyricWord] = [
            LyricWord(
                word=(w.get("word") or "").strip(),
                startTime=w.get("start", start),
                endTime=w.get("end", end),
                score=w.get("score", 1.0),
            )
            for w in seg.get("words", [])
            if (w.get("word") or "").strip()
        ]

        lines.append(TranscribedLine(
            index=len(lines),
            startTime=start,
            endTime=end,
            text=text,
            words=words,
        ))

    log(f"Post-processing: {len(lines)} lines kept, {dropped} dropped")
    if progress_cb:
        progress_cb(70)

    result = {
        "youtubeId": youtube_id,
        "lyrics":    [_line_to_dict(ln) for ln in lines],
        "metadata":  {
            "duration":     audio_duration,
            "language":     "ja",
            "whisperModel": whisper_model,
            "vocalsOnly":   vocals_only,
        },
    }
    lyrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    mark_stage_complete(youtube_id, "align", {
        "lyricsPath": "lyrics.json",
        "lineCount":  len(lines),
    })
    log(f"Wrote {len(lines)} lines to lyrics.json")

    if progress_cb:
        progress_cb(100)
    return lines
