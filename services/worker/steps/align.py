"""
Step 4 — Post-processing: convert raw WhisperX segments into
TranscribedLine objects, filter noise, and write lyrics.json.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

from models import LyricWord, TranscribedLine

logger = logging.getLogger(__name__)

# Segments shorter than this (seconds) or mostly silence are dropped
MIN_SEGMENT_DURATION = 0.2
# Lines with no recognisable Japanese characters are dropped
_JAPANESE_CHARS = set(
    "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめも"
    "やゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽ"
    "ぁぃぅぇぉゃゅょっアイウエオカキクケコサシスセソタチツテトナニヌネノ"
    "ハヒフヘホマミムメモヤユヨラリルレロワヲンガギグゲゴザジズゼゾダヂヅデド"
    "バビブベボパピプペポァィゥェォャュョッ"
)


def _has_japanese(text: str) -> bool:
    return any(c in _JAPANESE_CHARS for c in text)


def process_segments(
    raw_segments: list[dict],
    output_dir: Path,
    youtube_id: str,
    audio_duration: float,
    vocals_only: bool,
    whisper_model: str,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> list[TranscribedLine]:
    """
    Convert WhisperX output to clean TranscribedLine list and write lyrics.json.
    """
    lines: list[TranscribedLine] = []

    for idx, seg in enumerate(raw_segments):
        start = seg.get("start", 0.0)
        end   = seg.get("end",   start)
        text  = (seg.get("text") or "").strip()

        if end - start < MIN_SEGMENT_DURATION:
            continue
        if not _has_japanese(text):
            logger.debug("Dropping non-Japanese segment: %r", text)
            continue

        words: list[LyricWord] = []
        for w in seg.get("words", []):
            word_text = (w.get("word") or "").strip()
            if not word_text:
                continue
            words.append(
                LyricWord(
                    word=word_text,
                    startTime=w.get("start", start),
                    endTime=w.get("end", end),
                    score=w.get("score", 1.0),
                )
            )

        lines.append(
            TranscribedLine(
                index=len(lines),
                startTime=start,
                endTime=end,
                text=text,
                words=words,
            )
        )

    if progress_cb:
        progress_cb(80)

    # Serialise to lyrics.json
    result = {
        "youtubeId":   youtube_id,
        "lyrics":      [_line_to_dict(ln) for ln in lines],
        "metadata": {
            "duration":     audio_duration,
            "language":     "ja",
            "whisperModel": whisper_model,
            "vocalsOnly":   vocals_only,
        },
    }

    lyrics_path = output_dir / "lyrics.json"
    lyrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %d lines to %s", len(lines), lyrics_path)

    if progress_cb:
        progress_cb(100)

    return lines


def _line_to_dict(line: TranscribedLine) -> dict:
    return {
        "index":     line.index,
        "startTime": line.start_time,
        "endTime":   line.end_time,
        "text":      line.text,
        "words": [
            {
                "word":      w.word,
                "startTime": w.start_time,
                "endTime":   w.end_time,
                "score":     w.score,
            }
            for w in line.words
        ],
    }
