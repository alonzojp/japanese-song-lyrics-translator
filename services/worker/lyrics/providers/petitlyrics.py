"""
Provider 7 — PetitLyrics web scraping.

Japanese lyrics database with good J-Pop coverage.
Supports timed lyrics on some songs.
Priority 7, confidence 0.78.
"""
from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup

from lyrics.normalizer import normalize_text, text_to_lines
from lyrics.preprocessor import preprocess, compute_confidence_adjustment
from lyrics.types import LyricLine, LyricsResult, VideoInfo
from lyrics.providers.base import LyricsProvider

logger = logging.getLogger(__name__)

SEARCH_URL  = "https://petitlyrics.com/search_lyrics?title={title}&ct=2"
BASE_URL    = "https://petitlyrics.com"
_TIMEOUT    = 10.0
_HEADERS    = {"User-Agent": "Mozilla/5.0 (compatible; LyricsBot/1.0)"}
_MIN_LINES  = 4


class PetitLyricsProvider(LyricsProvider):
    name     = "petitlyrics"
    priority = 7

    async def fetch(self, video: VideoInfo) -> Optional[LyricsResult]:
        title = video.title or video.search_query
        if not title:
            return None

        async with httpx.AsyncClient(headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True) as client:
            song_url = await _search(client, title)
            if not song_url:
                return None
            raw_text, page_url = await _scrape(client, song_url)

        if not raw_text:
            return None

        norm  = normalize_text(raw_text)
        lines = [LyricLine(text=ln) for ln in text_to_lines(norm)]
        if len(lines) < _MIN_LINES:
            return None

        lines, stats = preprocess(lines)
        adj        = compute_confidence_adjustment(stats)
        confidence = round(0.78 * adj, 3)

        return LyricsResult(
            lines=lines,
            raw_text="\n".join(ln.text for ln in lines),
            provider="petitlyrics",
            confidence=confidence,
            has_timestamps=False,
            timestamp_quality=0.0,
            source_url=page_url or song_url,
            metadata={
                "title_query":      title,
                "line_count":       len(lines),
                "avg_ja_ratio":     stats.get("avg_japanese_ratio"),
                "preprocess_stats": stats,
            },
        )


async def _search(client: httpx.AsyncClient, title: str) -> Optional[str]:
    url = SEARCH_URL.format(title=quote(title))
    try:
        resp = await client.get(url)
        resp.raise_for_status()
    except Exception as exc:
        logger.debug("PetitLyrics search failed: %s", exc)
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    link = soup.select_one("a[href*='/lyrics/']")
    if not link:
        return None
    href = link.get("href", "")
    return href if href.startswith("http") else BASE_URL + href


async def _scrape(
    client: httpx.AsyncClient,
    url: str,
) -> tuple[Optional[str], Optional[str]]:
    try:
        resp = await client.get(url)
        resp.raise_for_status()
    except Exception as exc:
        logger.debug("PetitLyrics page fetch failed: %s", exc)
        return None, None

    soup  = BeautifulSoup(resp.text, "lxml")
    final = str(resp.url)

    # PetitLyrics main lyrics div
    area = (
        soup.find("div", class_="lyrics")
        or soup.find("div", id="lyrics")
        or soup.find("section", class_=lambda c: c and "lyric" in c.lower())
    )
    if not area:
        return None, final

    for br in area.find_all("br"):
        br.replace_with("\n")

    return area.get_text(separator="\n"), final
