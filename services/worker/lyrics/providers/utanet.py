"""
Provider 5 — Uta-net (uta-net.com) web scraping.

One of the most comprehensive Japanese lyrics databases.
No API required. Priority 5, confidence 0.88.
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

SEARCH_URL = "https://www.uta-net.com/search/?Keyword={query}&sort=0&target=2"
BASE_URL   = "https://www.uta-net.com"
_TIMEOUT   = 12.0
_HEADERS   = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "ja,en;q=0.9",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
_MIN_LINES = 4


class UtaNetProvider(LyricsProvider):
    name     = "utanet"
    priority = 2

    async def fetch(self, video: VideoInfo) -> Optional[LyricsResult]:
        # Try full query first, then title-only as fallback
        for query in _queries(video):
            result = await _try_query(video, query)
            if result:
                return result
        return None


def _queries(video: VideoInfo) -> list[str]:
    """
    Priority order for uta-net title search:
    1. song_title alone  — e.g. 'ときめきのソルフェージュ'  (most precise)
    2. search_query      — uploader + song_title
    3. title_only_query  — cleaned full title if song_title extraction failed
    """
    seen: list[str] = []
    for q in [video.song_title, video.search_query, video.title_only_query]:
        if q and q not in seen:
            seen.append(q)
    return seen


async def _try_query(video: VideoInfo, query: str) -> Optional[LyricsResult]:
    async with httpx.AsyncClient(headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True) as client:
        song_url = await _search(client, query)
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
    if len(lines) < _MIN_LINES:
        return None

    adj        = compute_confidence_adjustment(stats)
    confidence = round(0.88 * adj, 3)

    return LyricsResult(
        lines=lines,
        raw_text="\n".join(ln.text for ln in lines),
        provider="utanet",
        confidence=confidence,
        has_timestamps=False,
        timestamp_quality=0.0,
        source_url=page_url or song_url,
        metadata={
            "search_query": query,
            "line_count":   len(lines),
            "avg_ja_ratio": stats.get("avg_japanese_ratio"),
        },
    )


async def _search(client: httpx.AsyncClient, query: str) -> Optional[str]:
    url = SEARCH_URL.format(query=quote(query))
    try:
        resp = await client.get(url)
        resp.raise_for_status()
    except Exception as exc:
        logger.debug("Uta-net search failed: %s", exc)
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    # Song links are in a table; href pattern is /song/NNNNN/
    link = soup.select_one("a[href*='/song/']")
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
        logger.debug("Uta-net page fetch failed: %s", exc)
        return None, None

    soup  = BeautifulSoup(resp.text, "lxml")
    final = str(resp.url)

    area = (
        soup.find("div", id="kashi_area")
        or soup.find("div", class_="kashi_area")
        or soup.find("p", id="kashi_area")
        or soup.find("div", id="lyric-body")
        or soup.find("div", class_="lyric-body")
        or soup.find(id=lambda x: x and "kashi" in x.lower())
        or soup.find(class_=lambda x: x and "kashi" in x.lower())
        or soup.find(id=lambda x: x and "lyric" in x.lower())
        or soup.find(class_=lambda x: x and "lyric" in x.lower())
    )
    if not area:
        logger.debug("Uta-net: could not find lyrics container on %s", url)
        return None, final

    for br in area.find_all("br"):
        br.replace_with("\n")
    # Strip furigana ruby annotations
    for rt in area.find_all("rt"):
        rt.decompose()

    return area.get_text(separator="\n"), final
