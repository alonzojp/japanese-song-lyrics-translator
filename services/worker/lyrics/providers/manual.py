"""
Provider 8 — Manual upload / stored override.

Checks for a manually stored lyrics file: {CACHE_DIR}/{youtube_id}/lyrics_manual.txt
If present, uses it as lyrics (highest trust for user-provided content).

This file can be created by:
  POST /lyrics/{youtube_id}/manual  { text: "..." }

Priority 8 in the fallback chain, but can also act as an override
(the fetcher can be called with force=False and the manual file takes precedence).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from cache import cache_dir
from lyrics.normalizer import normalize_text, text_to_lines
from lyrics.preprocessor import preprocess, compute_confidence_adjustment
from lyrics.types import LyricLine, LyricsResult, VideoInfo
from lyrics.providers.base import LyricsProvider

logger = logging.getLogger(__name__)

MANUAL_FILE = "lyrics_manual.txt"
_MIN_LINES  = 2


class ManualProvider(LyricsProvider):
    name     = "manual"
    priority = 8

    async def fetch(self, video: VideoInfo) -> Optional[LyricsResult]:
        manual_path = cache_dir(video.youtube_id) / MANUAL_FILE
        if not manual_path.exists():
            return None

        raw = manual_path.read_text(encoding="utf-8").strip()
        if not raw:
            return None

        norm  = normalize_text(raw)
        lines = [LyricLine(text=ln) for ln in text_to_lines(norm)]
        if len(lines) < _MIN_LINES:
            return None

        lines, stats = preprocess(lines)
        # Manual upload gets high base confidence — user knows what they provided
        adj        = compute_confidence_adjustment(stats)
        confidence = round(0.90 * adj, 3)

        return LyricsResult(
            lines=lines,
            raw_text="\n".join(ln.text for ln in lines),
            provider="manual",
            confidence=confidence,
            has_timestamps=False,
            timestamp_quality=0.0,
            source_url=None,
            metadata={
                "line_count":       len(lines),
                "avg_ja_ratio":     stats.get("avg_japanese_ratio"),
                "preprocess_stats": stats,
                "manual_file":      str(manual_path),
            },
        )


def store_manual(youtube_id: str, text: str) -> Path:
    """Write user-provided lyrics text for a video."""
    path = cache_dir(youtube_id) / MANUAL_FILE
    path.write_text(text.strip(), encoding="utf-8")
    return path
