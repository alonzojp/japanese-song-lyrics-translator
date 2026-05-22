"""
Lyrics fetcher — chains all providers in priority order with full debug output.

Cache: stores result as {CACHE_DIR}/{youtube_id}/lyrics_cached.json
Re-fetching the same video returns the cached result instantly unless force=True.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from cache import cache_dir
from lyrics.types import LyricsResult, ProviderAttempt, VideoInfo
from lyrics.providers.youtube_captions import YouTubeCaptionsProvider
from lyrics.providers.youtube_auto_captions import YouTubeAutoCaptionsProvider
from lyrics.providers.utanet import UtaNetProvider
from lyrics.providers.utaten import UtatenProvider
from lyrics.providers.petitlyrics import PetitLyricsProvider
from lyrics.providers.genius import GeniusProvider
from lyrics.providers.description import DescriptionProvider
from lyrics.providers.comments import CommentsProvider
from lyrics.providers.manual import ManualProvider

logger = logging.getLogger(__name__)

CACHE_FILE = "lyrics_cached.json"

# Provider registry — ordered by priority (ascending = tried first)
# Accuracy-first: official captions → Japanese lyrics DBs → Genius → description/comments → auto-captions → manual
_PROVIDERS = [
    YouTubeCaptionsProvider(),      # priority 1 — official human-authored captions
    UtaNetProvider(),               # priority 2 — best Japanese lyrics DB
    UtatenProvider(),               # priority 3 — Japanese lyrics DB
    PetitLyricsProvider(),          # priority 4 — Japanese lyrics DB
    GeniusProvider(),               # priority 5
    DescriptionProvider(),          # priority 6
    CommentsProvider(),             # priority 7
    YouTubeAutoCaptionsProvider(),  # priority 8 — last resort, often inaccurate for music
    ManualProvider(),               # priority 9 — stored override
]


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_path(youtube_id: str) -> Path:
    return cache_dir(youtube_id) / CACHE_FILE


def get_cached(youtube_id: str) -> Optional[dict]:
    path = _cache_path(youtube_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def store_result(youtube_id: str, result: LyricsResult) -> None:
    data = result.to_dict()
    data["youtubeId"] = youtube_id
    _cache_path(youtube_id).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── Main fetch ─────────────────────────────────────────────────────────────────

async def fetch_lyrics(
    video: VideoInfo,
    force: bool = False,
    job_log=None,
) -> dict:
    """
    Fetch lyrics for a video, trying providers in priority order.

    Returns the serialized LyricsResult dict (same shape as the cache file).
    Always includes providerAttempts with debug info for every tried provider.
    """
    def log(msg: str) -> None:
        if job_log:
            job_log.info(msg, stage="lyrics")
        logger.info(msg)

    # ── Cache hit ──────────────────────────────────────────────────────────────
    if not force:
        cached = get_cached(video.youtube_id)
        if cached:
            log(
                f"Lyrics cache hit for {video.youtube_id} "
                f"(provider={cached.get('provider')}, "
                f"confidence={cached.get('confidence'):.2f})"
            )
            return cached

    log(f"Fetching lyrics for '{video.title or video.youtube_id}' …")

    all_attempts: list[ProviderAttempt] = []
    result: Optional[LyricsResult] = None

    for provider in sorted(_PROVIDERS, key=lambda p: p.priority):
        log(f"  Trying provider: {provider.name} (priority={provider.priority})")

        res, attempt = await provider.fetch_with_attempt(video)
        all_attempts.append(attempt)

        if res is not None:
            log(
                f"  ✓ {provider.name} succeeded — "
                f"{len(res.lines)} lines, "
                f"confidence={res.confidence:.2f}, "
                f"timestamps={'yes' if res.has_timestamps else 'no'}"
            )
            result = res
            break
        else:
            log(f"  ✗ {provider.name} failed: {attempt.failure_reason}")

    if result is None:
        log("All providers failed — no lyrics found")
        # Return an empty result with full debug
        from lyrics.types import LyricsResult as LR
        result = LR(
            lines=[],
            raw_text="",
            provider="none",
            confidence=0.0,
            has_timestamps=False,
            timestamp_quality=0.0,
            source_url=None,
            provider_attempts=all_attempts,
        )
    else:
        # Attach all attempts (including the failing ones before this provider)
        failing = [a for a in all_attempts if not a.succeeded]
        result.provider_attempts = failing + result.provider_attempts

    store_result(video.youtube_id, result)
    data = result.to_dict()
    data["youtubeId"] = video.youtube_id
    return data
