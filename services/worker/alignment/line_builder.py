"""
Build LyricLine objects from aligned segments.

Each LyricLine matches the TypeScript interface:
  {
    id:        str        e.g. "line-0"
    text:      str        clean lyric line text
    startTime: float
    endTime:   float
    words?:    WordTiming[]
    confidence?: float    per-line alignment confidence
  }

Merging strategy:
  - Very short segments (< MIN_CHARS) are merged into the next segment
  - Silence gaps > MERGE_GAP_THRESHOLD are treated as line breaks regardless
"""
from __future__ import annotations

import re
from typing import Optional

from lyrics.normalizer import normalize_line, is_noise_line

# Lines shorter than this are considered "continuation" candidates
MIN_CHARS          = 3
# Silence longer than this forces a new line (even if text would merge)
MERGE_GAP_THRESHOLD = 3.0   # seconds


def build_lyric_lines(
    segments:          list[dict],
    line_confidences:  Optional[list[float]] = None,
) -> list[dict]:
    """
    Convert aligned segments into LyricLine dicts ready for JSON serialisation.

    segments:  [{start, end, text, words: [{word, start, end, score}]}]
    Returns:   [{id, text, startTime, endTime, words: [{word, start, end}], confidence}]
    """
    merged    = _merge_short_segments(segments)
    result: list[dict] = []

    for i, seg in enumerate(merged):
        text = normalize_line(seg.get("text", ""))
        if is_noise_line(text):
            continue

        words = _build_word_timings(seg.get("words") or [])
        # If words span a wider range than segment, use word boundaries
        if words:
            start = min(words[0]["start"], seg.get("start", 0.0))
            end   = max(words[-1]["end"],  seg.get("end",   start + 1.0))
        else:
            start = seg.get("start", 0.0)
            end   = seg.get("end",   start + 1.0)

        conf = line_confidences[i] if line_confidences and i < len(line_confidences) else None

        entry: dict = {
            "id":        f"line-{len(result)}",
            "text":       text,
            "startTime":  round(start, 3),
            "endTime":    round(end, 3),
            "words":      words,
        }
        if conf is not None:
            entry["confidence"] = round(conf, 3)

        # Thread the VAD anchor offset through to visual_timing_normalization
        # so vocal onset can be recovered without separate RMS audio analysis.
        if seg.get('_vad_anchor_ms') is not None:
            entry['_vad_anchor_ms'] = seg['_vad_anchor_ms']

        result.append(entry)

    return result


def _build_word_timings(raw_words: list[dict]) -> list[dict]:
    """Convert raw word dicts to the public WordTiming format."""
    out: list[dict] = []
    for w in raw_words:
        word = (w.get("word") or "").strip()
        if not word:
            continue
        out.append({
            "word":  word,
            "start": round(float(w.get("start", 0.0)), 3),
            "end":   round(float(w.get("end",   0.0)), 3),
        })
    return out


def _merge_short_segments(segments: list[dict]) -> list[dict]:
    """
    Merge consecutive very-short segments unless separated by a long silence.
    """
    if not segments:
        return []

    result: list[dict] = [dict(segments[0])]

    for seg in segments[1:]:
        prev = result[-1]
        gap  = seg.get("start", 0.0) - prev.get("end", 0.0)
        text = (seg.get("text") or "").strip()

        if len(text) < MIN_CHARS and gap < MERGE_GAP_THRESHOLD:
            # Merge into previous
            prev["text"]  = (prev.get("text", "") + text).strip()
            prev["end"]   = seg.get("end", prev["end"])
            prev_words    = prev.get("words") or []
            seg_words     = seg.get("words")  or []
            prev["words"] = prev_words + seg_words
        else:
            result.append(dict(seg))

    return result
