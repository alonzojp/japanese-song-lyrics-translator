"""
Intermediate output file management.

Files written per video:
  transcript.json       raw Whisper segments (text only, no alignment)
  aligned_words.json    segments enriched with per-word timestamps
  aligned_lines.json    final LyricLine array (public output)
  alignment_meta.json   confidence report + processing metadata
  lyrics.json           alias of aligned_lines.json (backwards compatibility)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

TRANSCRIPT_FILE     = "transcript.json"
ALIGNED_WORDS_FILE  = "aligned_words.json"
ALIGNED_LINES_FILE  = "aligned_lines.json"
ALIGNMENT_META_FILE = "alignment_meta.json"
LYRICS_FILE         = "lyrics.json"

# Bump this string whenever alignment logic changes so stale aligned_words.json
# files are automatically rejected and force_align() re-runs.
ALIGNER_VERSION = "12"


def save_transcript(output_dir: Path, segments: list[dict], backend: str) -> Path:
    path = output_dir / TRANSCRIPT_FILE
    payload = {
        "backend":      backend,
        "segmentCount": len(segments),
        "savedAt":      _now(),
        "segments":     _strip_words(segments),   # raw text segments only
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_aligned_words(
    output_dir: Path,
    segments:   list[dict],
    method:     str,
) -> Path:
    path = output_dir / ALIGNED_WORDS_FILE
    payload = {
        "alignerVersion":  ALIGNER_VERSION,
        "alignmentMethod": method,
        "segmentCount":    len(segments),
        "savedAt":         _now(),
        "segments":        segments,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_aligned_lines(output_dir: Path, lines: list[dict]) -> Path:
    path = output_dir / ALIGNED_LINES_FILE
    path.write_text(
        json.dumps({"lineCount": len(lines), "savedAt": _now(), "lines": lines},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Also write the backwards-compatible lyrics.json
    lyrics_path = output_dir / LYRICS_FILE
    lyrics_path.write_text(
        json.dumps(
            {
                "youtubeId": output_dir.name,
                "lyrics":    _lines_to_legacy(lines),
                "metadata":  {"savedAt": _now()},
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    return path


def save_alignment_meta(
    output_dir: Path,
    youtube_id: str,
    backend:    str,
    method:     str,
    confidence: dict,
    audio_duration: float,
    whisper_model:  str,
    vocals_only:    bool,
) -> Path:
    path = output_dir / ALIGNMENT_META_FILE
    payload = {
        "youtubeId":      youtube_id,
        "transcriptionBackend": backend,
        "alignmentMethod":      method,
        "whisperModel":         whisper_model,
        "audioDetails": {
            "duration":    audio_duration,
            "language":    "ja",
            "vocalsOnly":  vocals_only,
        },
        "confidence": confidence,
        "processedAt": _now(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ── Load helpers ───────────────────────────────────────────────────────────────

def load_transcript(output_dir: Path) -> Optional[list[dict]]:
    path = output_dir / TRANSCRIPT_FILE
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("segments")


def load_aligned_words(output_dir: Path) -> Optional[tuple[list[dict], str]]:
    path = output_dir / ALIGNED_WORDS_FILE
    if not path.exists():
        return None
    data    = json.loads(path.read_text(encoding="utf-8"))
    version = data.get("alignerVersion", "0")
    if version != ALIGNER_VERSION:
        import logging
        logging.getLogger(__name__).info(
            f"aligned_words.json version mismatch ({version} != {ALIGNER_VERSION}) "
            f"at {path} — discarding cache, will re-align"
        )
        return None
    return data.get("segments"), data.get("alignmentMethod", "unknown")


def load_aligned_lines(output_dir: Path) -> Optional[list[dict]]:
    path = output_dir / ALIGNED_LINES_FILE
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("lines")


def load_alignment_meta(output_dir: Path) -> Optional[dict]:
    path = output_dir / ALIGNMENT_META_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_words(segments: list[dict]) -> list[dict]:
    """Return segments without the words array (for transcript.json readability)."""
    return [
        {k: v for k, v in seg.items() if k != "words"}
        for seg in segments
    ]


def _lines_to_legacy(lines: list[dict]) -> list[dict]:
    """Convert LyricLine format to the legacy TranscribedLine format."""
    return [
        {
            "index":     i,
            "startTime": ln["startTime"],
            "endTime":   ln["endTime"],
            "text":      ln["text"],
            "words":     [
                {"word": w["word"], "startTime": w["start"], "endTime": w["end"], "score": 1.0}
                for w in ln.get("words", [])
            ],
        }
        for i, ln in enumerate(lines)
    ]
