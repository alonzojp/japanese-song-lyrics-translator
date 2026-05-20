"""
Shared types for the lyrics provider system.
Mirrors packages/shared/src/api/lyrics.ts — keep in sync.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class VideoInfo:
    """Everything we know about the video when querying providers."""
    youtube_id:  str
    youtube_url: str
    title:       Optional[str] = None
    uploader:    Optional[str] = None
    description: Optional[str] = None
    duration:    Optional[float] = None

    @property
    def search_query(self) -> str:
        parts = []
        if self.uploader:
            parts.append(self.uploader)
        if self.title:
            parts.append(self.title)
        return " ".join(parts) if parts else self.youtube_id


@dataclass
class LyricLine:
    text:       str
    start_time: Optional[float] = None   # seconds
    end_time:   Optional[float] = None


@dataclass
class ProviderAttempt:
    provider:       str
    priority:       int
    succeeded:      bool
    failure_reason: Optional[str] = None
    duration_ms:    int = 0
    debug_notes:    list[str] = field(default_factory=list)


@dataclass
class LyricsResult:
    lines:             list[LyricLine]
    raw_text:          str
    provider:          str
    confidence:        float          # 0.0–1.0
    has_timestamps:    bool
    timestamp_quality: float          # 0.0–1.0 (0 if no timestamps)
    source_url:        Optional[str]
    metadata: dict = field(default_factory=dict)
    # Debug: all attempts (successful and failed)
    provider_attempts: list[ProviderAttempt] = field(default_factory=list)
    cached_at:         str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "provider":         self.provider,
            "confidence":       round(self.confidence, 3),
            "hasTimestamps":    self.has_timestamps,
            "timestampQuality": round(self.timestamp_quality, 3),
            "sourceUrl":        self.source_url,
            "metadata":         self.metadata,
            "cachedAt":         self.cached_at,
            "lines": [
                {
                    "text":      ln.text,
                    "startTime": ln.start_time,
                    "endTime":   ln.end_time,
                }
                for ln in self.lines
            ],
            "providerAttempts": [
                {
                    "provider":      a.provider,
                    "priority":      a.priority,
                    "succeeded":     a.succeeded,
                    "failureReason": a.failure_reason,
                    "durationMs":    a.duration_ms,
                    "debugNotes":    a.debug_notes,
                }
                for a in self.provider_attempts
            ],
            "rawText": self.raw_text,
        }
