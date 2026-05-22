"""
Provider 3 — YouTube description scraping.

Some official channels (Vocaloid labels, anime studios) paste full lyrics
in the video description. Priority 3, confidence 0.50 (no timestamps).
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import yt_dlp

from lyrics.normalizer import (
    normalize_text,
    text_to_lines,
    japanese_char_ratio,
)
from lyrics.preprocessor import preprocess, compute_confidence_adjustment
from lyrics.types import LyricLine, LyricsResult, VideoInfo
from lyrics.providers.base import LyricsProvider

logger = logging.getLogger(__name__)

# Lyrics blocks in descriptions are usually preceded by a header line
_LYRICS_HEADER_RE = re.compile(
    r"(?:▶\s*lyrics?|▶\s*lyric|lyrics?|歌詞|作詞|lyric|words?)[：:＝=\s]*\n",
    re.IGNORECASE,
)

# Lines that look like credits (role: person) — used to reject credits blocks
_CREDITS_LINE_RE = re.compile(r"^.{1,15}[：:].{1,30}$")

# Minimum Japanese character ratio to consider a block as lyrics
_MIN_JA_RATIO = 0.35
# Minimum lines to accept as lyrics
_MIN_LINES = 4


class DescriptionProvider(LyricsProvider):
    name     = "description"
    priority = 6

    async def fetch(self, video: VideoInfo) -> Optional[LyricsResult]:
        description = video.description
        if not description:
            # Fetch via yt-dlp (no download)
            description = await _fetch_description(video.youtube_url)
        if not description:
            return None

        # Try to find a lyrics block
        block = _extract_lyrics_block(description)
        if not block:
            return None

        text  = normalize_text(block)
        lines = [LyricLine(text=ln) for ln in text_to_lines(text)]

        if len(lines) < _MIN_LINES:
            return None

        lines, stats = preprocess(lines)
        ja_ratio = stats.get("avg_japanese_ratio", 0.0)
        if ja_ratio < _MIN_JA_RATIO or len(lines) < _MIN_LINES:
            return None

        adj        = compute_confidence_adjustment(stats)
        confidence = round(0.50 * adj, 3)

        return LyricsResult(
            lines=lines,
            raw_text="\n".join(ln.text for ln in lines),
            provider="description",
            confidence=confidence,
            has_timestamps=False,
            timestamp_quality=0.0,
            source_url=video.youtube_url,
            metadata={
                "line_count":       len(lines),
                "avg_ja_ratio":     ja_ratio,
                "preprocess_stats": stats,
            },
        )


async def _fetch_description(youtube_url: str) -> Optional[str]:
    try:
        ydl_opts = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False) or {}
        return info.get("description")
    except Exception:
        return None


def _extract_lyrics_block(description: str) -> Optional[str]:
    """
    Heuristic: find the most Japanese-dense paragraph in the description.
    If a lyrics header is present, use everything after it.
    """
    # Try header-based extraction first
    m = _LYRICS_HEADER_RE.search(description)
    if m:
        block = description[m.end():]
        # Stop at the next section header (a line that looks like a label)
        end_m = re.search(r"\n[A-Za-z　-鿿]{1,20}[：:＝=]\s*\n", block)
        if end_m:
            block = block[: end_m.start()]
        return block.strip()

    # Fallback: split into paragraphs, return the one with highest Japanese density
    # that doesn't look like credits (role: person format)
    paragraphs = re.split(r"\n{2,}", description.strip())
    best_block: Optional[str] = None
    best_ratio  = 0.0
    for para in paragraphs:
        lines = para.splitlines()
        # Skip blocks where most lines look like credits
        credits_count = sum(1 for ln in lines if _CREDITS_LINE_RE.match(ln.strip()))
        if len(lines) > 0 and credits_count / len(lines) > 0.5:
            continue
        ratio = japanese_char_ratio(para)
        if ratio > best_ratio and ratio > _MIN_JA_RATIO and len(lines) >= _MIN_LINES:
            best_ratio = ratio
            best_block = para

    return best_block
