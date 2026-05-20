"""
Abstract base class for all lyrics providers.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

from lyrics.types import LyricsResult, ProviderAttempt, VideoInfo


class LyricsProvider(ABC):
    name:     str
    priority: int   # lower = tried first

    @abstractmethod
    async def fetch(self, video: VideoInfo) -> Optional[LyricsResult]:
        """
        Attempt to fetch lyrics for the given video.
        Return None (do not raise) if this provider cannot serve the request.
        Raise only for unexpected runtime errors.
        """
        ...

    async def fetch_with_attempt(
        self,
        video: VideoInfo,
    ) -> tuple[Optional[LyricsResult], ProviderAttempt]:
        """
        Wraps fetch() to record timing + success/failure metadata.
        The caller adds the returned ProviderAttempt to the running list.
        """
        start = time.monotonic()
        attempt = ProviderAttempt(
            provider=self.name,
            priority=self.priority,
            succeeded=False,
        )
        try:
            result = await self.fetch(video)
            elapsed = int((time.monotonic() - start) * 1000)
            attempt.duration_ms = elapsed
            if result is not None:
                attempt.succeeded = True
                # Copy provider_attempts list so callers can prepend it
                result.provider_attempts = [attempt]
            else:
                attempt.failure_reason = "provider returned None"
            return result, attempt
        except Exception as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            attempt.duration_ms    = elapsed
            attempt.failure_reason = str(exc)[:300]
            return None, attempt
