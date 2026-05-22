"""
Provider 6 — Utaten (うたてん) web scraping.

Japanese lyrics database, excellent coverage of J-Pop / anime.
No API required. Priority 6, confidence 0.80.
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

UTATEN_SEARCH = "https://utaten.com/search?q={query}&target=title_artist"
UTATEN_BASE   = "https://utaten.com"
_TIMEOUT      = 10.0
_HEADERS      = {"User-Agent": "Mozilla/5.0 (compatible; LyricsBot/1.0)"}
_MIN_LINES    = 4


class UtatenProvider(LyricsProvider):
    name     = "utaten"
    priority = 3

    async def fetch(self, video: VideoInfo) -> Optional[LyricsResult]:
        # Try full query then title-only fallback
        for query in dict.fromkeys([video.search_query, video.title_only_query]):
            if not query:
                continue
            async with httpx.AsyncClient(headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True) as client:
                song_url = await _search(client, query)
                if not song_url:
                    continue
                raw_text, page_url = await _scrape(client, song_url)
            if raw_text:
                break
        else:
            return None

        query = video.search_query  # keep for metadata

        if not raw_text:
            return None

        norm  = normalize_text(raw_text)
        lines = [LyricLine(text=ln) for ln in text_to_lines(norm)]
        if len(lines) < _MIN_LINES:
            return None

        lines, stats = preprocess(lines)
        adj        = compute_confidence_adjustment(stats)
        confidence = round(0.80 * adj, 3)

        return LyricsResult(
            lines=lines,
            raw_text="\n".join(ln.text for ln in lines),
            provider="utaten",
            confidence=confidence,
            has_timestamps=False,
            timestamp_quality=0.0,
            source_url=page_url or song_url,
            metadata={
                "search_query":     query,
                "line_count":       len(lines),
                "avg_ja_ratio":     stats.get("avg_japanese_ratio"),
                "preprocess_stats": stats,
            },
        )


async def _search(client: httpx.AsyncClient, query: str) -> Optional[str]:
    url = UTATEN_SEARCH.format(query=quote(query))
    try:
        resp = await client.get(url)
        resp.raise_for_status()
    except Exception as exc:
        logger.debug("Utaten search failed: %s", exc)
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    # Search result links contain /lyric/ in the href
    link = soup.select_one("a[href*='/lyric/']")
    if not link:
        return None
    href = link.get("href", "")
    return href if href.startswith("http") else UTATEN_BASE + href


async def _scrape(
    client: httpx.AsyncClient,
    url: str,
) -> tuple[Optional[str], Optional[str]]:
    try:
        resp = await client.get(url)
        resp.raise_for_status()
    except Exception as exc:
        logger.debug("Utaten page fetch failed: %s", exc)
        return None, None

    soup  = BeautifulSoup(resp.text, "lxml")
    final = str(resp.url)

    # Lyrics live in #kashi_area or .hiragana span elements
    area = soup.find(id="kashi_area") or soup.find("div", class_="lyrics")
    if not area:
        # Fallback: any element with many hiragana spans
        area = soup.find("div", class_=lambda c: c and "lyric" in c.lower())
    if not area:
        return None, final

    for br in area.find_all("br"):
        br.replace_with("\n")
    # Remove ruby annotation text (furigana) — keep only kanji surface
    for ruby in area.find_all("ruby"):
        # Keep just the base text (before <rt>)
        rt = ruby.find("rt")
        if rt:
            rt.decompose()

    return area.get_text(separator="\n"), final
