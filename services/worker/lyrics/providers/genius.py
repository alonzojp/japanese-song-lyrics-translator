"""
Provider 5 — Genius lyrics.

Uses the official Genius API for search (if GENIUS_ACCESS_TOKEN is set),
then scrapes the lyrics page (Genius API doesn't return full lyrics).
Priority 5, confidence 0.85.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

from lyrics.normalizer import normalize_text, text_to_lines, japanese_char_ratio
from lyrics.preprocessor import preprocess, compute_confidence_adjustment
from lyrics.types import LyricLine, LyricsResult, VideoInfo
from lyrics.providers.base import LyricsProvider

logger = logging.getLogger(__name__)

GENIUS_API_SEARCH = "https://api.genius.com/search"
GENIUS_SEARCH_URL = "https://genius.com/search?q={query}"
GENIUS_BASE       = "https://genius.com"
_TIMEOUT          = 10.0
_HEADERS          = {
    "User-Agent":      "Mozilla/5.0 (compatible; LyricsBot/1.0)",
    "Accept-Language": "ja,en;q=0.9",
}
_MIN_LINES    = 4
_MIN_JA_RATIO = 0.30
_GENIUS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN", "")


class GeniusProvider(LyricsProvider):
    name     = "genius"
    priority = 5

    async def fetch(self, video: VideoInfo) -> Optional[LyricsResult]:
        query = video.search_query
        if not query:
            return None

        async with httpx.AsyncClient(headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True) as client:
            # Step 1: search
            song_url = await _search(client, query, video)
            if not song_url:
                return None

            # Step 2: scrape lyrics page
            raw_text, page_url = await _scrape_lyrics(client, song_url)

        if not raw_text:
            return None

        norm  = normalize_text(raw_text)
        lines = [LyricLine(text=ln) for ln in text_to_lines(norm)]
        if len(lines) < _MIN_LINES:
            return None

        lines, stats = preprocess(lines)
        ja_ratio = stats.get("avg_japanese_ratio", 0.0)
        if ja_ratio < _MIN_JA_RATIO or len(lines) < _MIN_LINES:
            return None

        adj        = compute_confidence_adjustment(stats)
        confidence = round(0.85 * adj, 3)

        return LyricsResult(
            lines=lines,
            raw_text="\n".join(ln.text for ln in lines),
            provider="genius",
            confidence=confidence,
            has_timestamps=False,
            timestamp_quality=0.0,
            source_url=page_url or song_url,
            metadata={
                "search_query":     query,
                "line_count":       len(lines),
                "avg_ja_ratio":     ja_ratio,
                "preprocess_stats": stats,
            },
        )


async def _search(
    client: httpx.AsyncClient,
    query: str,
    video: VideoInfo,
) -> Optional[str]:
    # Prefer API search when token is available — much more reliable
    if _GENIUS_TOKEN:
        try:
            resp = await client.get(
                GENIUS_API_SEARCH,
                params={"q": query},
                headers={**_HEADERS, "Authorization": f"Bearer {_GENIUS_TOKEN}"},
            )
            resp.raise_for_status()
            hits = resp.json().get("response", {}).get("hits", [])
            for hit in hits:
                result = hit.get("result", {})
                url = result.get("url", "")
                if url:
                    return url
        except Exception as exc:
            logger.debug("Genius API search failed, falling back to scrape: %s", exc)

    # Fallback: scrape the search page
    url = GENIUS_SEARCH_URL.format(query=quote_plus(query))
    try:
        resp = await client.get(url)
        resp.raise_for_status()
    except Exception as exc:
        logger.debug("Genius search failed: %s", exc)
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    for link in soup.select("a[href*='/lyrics/'], a[href*='-lyrics']"):
        href = link.get("href", "")
        if not href.startswith("/") and not href.startswith(GENIUS_BASE):
            continue
        full = href if href.startswith("http") else GENIUS_BASE + href
        if "/lyrics/" in full or full.count("-") >= 2:
            return full

    return None


async def _scrape_lyrics(
    client: httpx.AsyncClient,
    url: str,
) -> tuple[Optional[str], Optional[str]]:
    try:
        resp = await client.get(url)
        resp.raise_for_status()
    except Exception as exc:
        logger.debug("Genius page fetch failed: %s", exc)
        return None, None

    soup  = BeautifulSoup(resp.text, "lxml")
    final = str(resp.url)

    # Genius lyrics containers use data-lyrics-container attribute
    containers = soup.find_all(attrs={"data-lyrics-container": "true"})
    if not containers:
        # Older pages: class contains "Lyrics__Container"
        containers = soup.find_all("div", class_=re.compile(r"Lyrics__Container"))

    if not containers:
        return None, final

    parts: list[str] = []
    for container in containers:
        # Replace <br> with newlines before extracting text
        for br in container.find_all("br"):
            br.replace_with("\n")
        text = container.get_text(separator="\n")
        parts.append(text)

    return "\n".join(parts), final
