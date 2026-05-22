"""
Provider 8 — YouTube auto-generated captions (last resort).

Only used when official captions, Japanese lyrics databases, Genius,
description, and comments all fail. Auto-generated Japanese captions
from YouTube's speech recognition are often inaccurate for music.
Priority 8, confidence 0.45.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

import yt_dlp

from lyrics.normalizer import normalize_line, is_noise_line
from lyrics.preprocessor import preprocess, compute_confidence_adjustment
from lyrics.providers.youtube_captions import (
    _parse_json3,
    _parse_vtt,
    _timestamp_quality,
)
from lyrics.types import LyricsResult, VideoInfo
from lyrics.providers.base import LyricsProvider

logger = logging.getLogger(__name__)


class YouTubeAutoCaptionsProvider(LyricsProvider):
    name     = "youtube_auto_captions"
    priority = 8

    async def fetch(self, video: VideoInfo) -> Optional[LyricsResult]:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ydl_opts = {
                "writesubtitles":    False,
                "writeautomaticsub": True,
                "subtitleslangs":    ["ja", "ja-JP", "ja-Hans"],
                "subtitlesformat":   "json3/vtt/best",
                "skip_download":     True,
                "quiet":             True,
                "no_warnings":       True,
                "outtmpl":           str(tmp_path / "subs.%(ext)s"),
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video.youtube_url, download=True) or {}

            # Bail if official captions exist — YouTubeCaptionsProvider handles those
            has_official = bool(info.get("subtitles", {}).get("ja") or
                                info.get("subtitles", {}).get("ja-JP"))
            if has_official:
                return None

            sub_files = sorted(tmp_path.glob("subs.*"))
            if not sub_files:
                return None

            sub_file    = sub_files[0]
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
        confidence = round(0.45 * adj, 3)

        return LyricsResult(
            lines=lines,
            raw_text="\n".join(ln.text for ln in lines),
            provider="youtube_auto_captions",
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
