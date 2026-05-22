"""
Provider 1 — YouTube official captions only.

Uses yt-dlp to download subtitle files (no video download).
Only fetches human-authored subtitles; auto-generated captions are handled
by YouTubeAutoCaptionsProvider at much lower priority.
Official captions → priority 1, confidence 0.95

Parses VTT and JSON3 formats including word-level timestamps from JSON3.
"""
from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Optional

import yt_dlp

from lyrics.normalizer import normalize_line, is_noise_line, japanese_char_ratio
from lyrics.preprocessor import preprocess, compute_confidence_adjustment
from lyrics.types import LyricLine, LyricsResult, VideoInfo
from lyrics.providers.base import LyricsProvider

logger = logging.getLogger(__name__)

# VTT timestamp: 00:01:23.456
_VTT_TS   = re.compile(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})")
# VTT cue header: 00:00:00.000 --> 00:00:03.000
_VTT_CUE  = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2}\.\d{3})")
# VTT HTML tags: <00:00:03.500><c>text</c>
_VTT_TAGS = re.compile(r"<[^>]+>")


def _ts_to_sec(ts: str) -> float:
    m = _VTT_TS.match(ts.strip())
    if not m:
        return 0.0
    h, mi, s, ms = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    return h * 3600 + mi * 60 + s + ms / 1000


def _parse_vtt(text: str) -> list[LyricLine]:
    """Parse WebVTT subtitle file into timed lines."""
    lines: list[LyricLine] = []
    blocks = re.split(r"\n\n+", text.strip())

    for block in blocks:
        block = block.strip()
        if not block or block.startswith("WEBVTT") or block.startswith("NOTE"):
            continue
        block_lines = block.splitlines()
        cue_line = next((ln for ln in block_lines if " --> " in ln), None)
        if not cue_line:
            continue
        m = _VTT_CUE.search(cue_line)
        if not m:
            continue
        start = _ts_to_sec(m.group(1))
        end   = _ts_to_sec(m.group(2))

        text_lines = [
            _VTT_TAGS.sub("", ln).strip()
            for ln in block_lines
            if " --> " not in ln and not ln.strip().isdigit()
        ]
        text = " ".join(ln for ln in text_lines if ln)
        text = normalize_line(text)
        if text and not is_noise_line(text):
            lines.append(LyricLine(text=text, start_time=start, end_time=end))

    return _dedupe_adjacent(lines)


def _parse_json3(data: dict) -> list[LyricLine]:
    """Parse YouTube's JSON3 caption format (includes word-level timing)."""
    lines: list[LyricLine] = []
    for event in data.get("events", []):
        start_ms = event.get("tStartMs", 0)
        dur_ms   = event.get("dDurationMs", 0)
        segs     = event.get("segs", [])
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs).replace("\n", " ").strip()
        text = normalize_line(text)
        if text and not is_noise_line(text):
            lines.append(LyricLine(
                text=text,
                start_time=start_ms / 1000,
                end_time=(start_ms + dur_ms) / 1000,
            ))
    return _dedupe_adjacent(lines)


def _dedupe_adjacent(lines: list[LyricLine]) -> list[LyricLine]:
    """Remove consecutive duplicate lines (auto-captions often repeat)."""
    result: list[LyricLine] = []
    for ln in lines:
        if result and result[-1].text == ln.text:
            # Extend end time
            result[-1] = LyricLine(
                text=result[-1].text,
                start_time=result[-1].start_time,
                end_time=ln.end_time or result[-1].end_time,
            )
        else:
            result.append(ln)
    return result


def _timestamp_quality(lines: list[LyricLine]) -> float:
    """0–1 quality score for timestamp coverage."""
    if not lines:
        return 0.0
    timed = sum(1 for ln in lines if ln.start_time is not None)
    return timed / len(lines)


class YouTubeCaptionsProvider(LyricsProvider):
    name     = "youtube_captions"
    priority = 1

    async def fetch(self, video: VideoInfo) -> Optional[LyricsResult]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ydl_opts = {
                "writesubtitles":    True,
                "writeautomaticsub": False,  # official only — auto-captions handled separately
                "subtitleslangs":    ["ja", "ja-JP", "ja-Hans"],
                "subtitlesformat":   "json3/vtt/best",
                "skip_download":     True,
                "quiet":             True,
                "no_warnings":       True,
                "outtmpl":           str(tmp_path / "subs.%(ext)s"),
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video.youtube_url, download=True) or {}

            has_official = bool(info.get("subtitles", {}).get("ja") or
                                info.get("subtitles", {}).get("ja-JP"))
            if not has_official:
                return None

            sub_files = sorted(tmp_path.glob("subs.*"))
            if not sub_files:
                return None

            sub_file = sub_files[0]
            raw_content = sub_file.read_text(encoding="utf-8", errors="replace")

            if sub_file.suffix == ".json3" or sub_file.name.endswith(".json3"):
                try:
                    data  = json.loads(raw_content)
                    lines = _parse_json3(data)
                except Exception:
                    lines = _parse_vtt(raw_content)
            else:
                lines = _parse_vtt(raw_content)

        if not lines:
            return None

        lines, stats = preprocess(lines)
        if not lines:
            return None

        adj        = compute_confidence_adjustment(stats)
        ts_quality = _timestamp_quality(lines)
        confidence = round(0.95 * adj, 3)

        return LyricsResult(
            lines=lines,
            raw_text="\n".join(ln.text for ln in lines),
            provider="youtube_captions",
            confidence=confidence,
            has_timestamps=True,
            timestamp_quality=ts_quality,
            source_url=video.youtube_url,
            metadata={
                "line_count":       len(lines),
                "avg_ja_ratio":     stats.get("avg_japanese_ratio"),
                "preprocess_stats": stats,
            },
        )
