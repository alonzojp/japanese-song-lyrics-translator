"""
Provider 4 — YouTube pinned/top comment scraping.

Some channels pin a comment with full lyrics.
Uses yt-dlp comment extraction (first page only — no deep pagination).
Priority 4, confidence 0.60.
"""
from __future__ import annotations

import logging
from typing import Optional

import yt_dlp

from lyrics.normalizer import normalize_text, text_to_lines, japanese_char_ratio
from lyrics.preprocessor import preprocess, compute_confidence_adjustment
from lyrics.types import LyricLine, LyricsResult, VideoInfo
from lyrics.providers.base import LyricsProvider

logger = logging.getLogger(__name__)

_MIN_LINES    = 6
_MIN_JA_RATIO = 0.40
# Lyrics comments are usually long
_MIN_CHARS    = 150


class CommentsProvider(LyricsProvider):
    name     = "comments"
    priority = 7

    async def fetch(self, video: VideoInfo) -> Optional[LyricsResult]:
        comments = await _fetch_top_comments(video.youtube_url)
        if not comments:
            return None

        # Find the longest Japanese-heavy comment (likely to be lyrics)
        best_text: Optional[str] = None
        best_ratio  = 0.0

        for comment in comments:
            text = comment.get("text", "")
            if len(text) < _MIN_CHARS:
                continue
            ratio = japanese_char_ratio(text)
            if ratio > best_ratio and ratio >= _MIN_JA_RATIO:
                best_ratio = ratio
                best_text  = text

        if not best_text:
            return None

        norm  = normalize_text(best_text)
        lines = [LyricLine(text=ln) for ln in text_to_lines(norm)]
        if len(lines) < _MIN_LINES:
            return None

        lines, stats = preprocess(lines)
        if len(lines) < _MIN_LINES:
            return None

        adj        = compute_confidence_adjustment(stats)
        confidence = round(0.60 * adj, 3)

        return LyricsResult(
            lines=lines,
            raw_text="\n".join(ln.text for ln in lines),
            provider="comments",
            confidence=confidence,
            has_timestamps=False,
            timestamp_quality=0.0,
            source_url=video.youtube_url,
            metadata={
                "line_count":       len(lines),
                "avg_ja_ratio":     best_ratio,
                "preprocess_stats": stats,
            },
        )


async def _fetch_top_comments(youtube_url: str, max_comments: int = 20) -> list[dict]:
    """Fetch first page of comments via yt-dlp (lightweight)."""
    try:
        ydl_opts = {
            "skip_download":   True,
            "quiet":           True,
            "no_warnings":     True,
            "getcomments":     True,
            "extractor_args": {
                "youtube": {
                    "max_comments": [str(max_comments)],
                    "comment_sort": ["top"],
                }
            },
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False) or {}
        return info.get("comments", []) or []
    except Exception as exc:
        logger.debug("Comment fetch failed: %s", exc)
        return []
