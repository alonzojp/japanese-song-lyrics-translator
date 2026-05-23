"""
Filesystem cache manager.

Layout per video:
  {CACHE_DIR}/{youtube_id}/
    manifest.json        — stage completion + metadata
    audio_raw.{ext}      — original yt-dlp download
    audio.wav            — normalised 44.1 kHz stereo  (Demucs input)
    vocals.wav           — Demucs vocals stem
    lyrics.json          — final WhisperX output

Cache key = YouTube video ID (11-char alphanumeric, filesystem-safe).
All timestamps are UTC ISO-8601.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config import CACHE_DIR

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"

# Stages in pipeline order
STAGES: list[str] = ["download", "separate", "transcribe", "align"]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Public API ─────────────────────────────────────────────────────────────────

def cache_dir(youtube_id: str) -> Path:
    """Return (and create) the cache directory for a video."""
    d = CACHE_DIR / youtube_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_manifest(youtube_id: str) -> dict:
    path = cache_dir(youtube_id) / MANIFEST_FILE
    return _load(path)


def _touch_manifest(youtube_id: str) -> dict:
    """Load manifest, updating lastAccessedAt."""
    manifest = get_manifest(youtube_id)
    manifest["lastAccessedAt"] = _now_iso()
    _save(cache_dir(youtube_id) / MANIFEST_FILE, manifest)
    return manifest


def is_stage_complete(youtube_id: str, stage: str) -> bool:
    """True if this stage was previously completed and its output file exists."""
    manifest = get_manifest(youtube_id)
    stage_data = manifest.get("stages", {}).get(stage, {})
    if not stage_data.get("completedAt"):
        return False

    # Verify the output file still exists
    output_key = {
        "download":   "audioPath",
        "separate":   "vocalsPath",
        "transcribe": "transcriptPath",
        "align":      "lyricsPath",
    }.get(stage)

    if output_key:
        rel_path = stage_data.get(output_key)
        if rel_path and not (cache_dir(youtube_id) / rel_path).exists():
            logger.warning(
                "Cache manifest claims %s/%s is done but file %s is missing — will reprocess",
                youtube_id, stage, rel_path,
            )
            return False

    return True


def get_stage_data(youtube_id: str, stage: str) -> dict:
    return get_manifest(youtube_id).get("stages", {}).get(stage, {})


def mark_stage_complete(youtube_id: str, stage: str, data: dict) -> None:
    """Record a completed stage in the manifest."""
    manifest = get_manifest(youtube_id)
    manifest.setdefault("stages", {})[stage] = {
        **data,
        "completedAt": _now_iso(),
    }
    manifest["lastAccessedAt"] = _now_iso()
    if "createdAt" not in manifest:
        manifest["createdAt"] = _now_iso()
    _save(cache_dir(youtube_id) / MANIFEST_FILE, manifest)


def store_metadata(youtube_id: str, metadata: dict) -> None:
    """Save video metadata (title, uploader, duration, thumbnail)."""
    manifest = get_manifest(youtube_id)
    manifest["metadata"] = metadata
    manifest["lastAccessedAt"] = _now_iso()
    if "createdAt" not in manifest:
        manifest["createdAt"] = _now_iso()
    _save(cache_dir(youtube_id) / MANIFEST_FILE, manifest)


def get_metadata(youtube_id: str) -> dict:
    return get_manifest(youtube_id).get("metadata", {})


def invalidate_stage(youtube_id: str, stage: str) -> None:
    """Remove a stage from the manifest so it re-runs on the next job."""
    manifest = get_manifest(youtube_id)
    manifest.get("stages", {}).pop(stage, None)
    _save(cache_dir(youtube_id) / MANIFEST_FILE, manifest)


def all_stages_complete(youtube_id: str) -> bool:
    return all(is_stage_complete(youtube_id, s) for s in STAGES)


# ── Cleanup ────────────────────────────────────────────────────────────────────

def list_cached() -> list[dict]:
    """Return summary of all cached videos."""
    results = []
    for d in sorted(CACHE_DIR.iterdir()):
        if not d.is_dir():
            continue
        manifest = _load(d / MANIFEST_FILE)
        stages_done = [s for s in STAGES if manifest.get("stages", {}).get(s, {}).get("completedAt")]
        results.append({
            "youtubeId":       d.name,
            "createdAt":       manifest.get("createdAt"),
            "lastAccessedAt":  manifest.get("lastAccessedAt"),
            "metadata":        manifest.get("metadata", {}),
            "stagesComplete":  stages_done,
            "fullyProcessed":  len(stages_done) == len(STAGES),
            "dirSizeMB":       _dir_size_mb(d),
        })
    return results


def cleanup_old(max_age_days: int = 30) -> list[str]:
    """
    Delete cache dirs that haven't been accessed in max_age_days days.
    Returns list of deleted youtube_ids.
    """
    cutoff = datetime.now(timezone.utc).timestamp() - max_age_days * 86400
    deleted: list[str] = []

    for d in CACHE_DIR.iterdir():
        if not d.is_dir():
            continue
        manifest = _load(d / MANIFEST_FILE)
        last_access = manifest.get("lastAccessedAt") or manifest.get("createdAt")
        if last_access:
            try:
                ts = datetime.fromisoformat(last_access).timestamp()
                if ts < cutoff:
                    shutil.rmtree(d)
                    deleted.append(d.name)
                    logger.info("Cache evicted (age > %d days): %s", max_age_days, d.name)
            except Exception as exc:
                logger.warning("Could not evaluate age for %s: %s", d.name, exc)

    return deleted


def cleanup_temp_files(youtube_id: str) -> None:
    """Remove intermediate files that are no longer needed after full processing."""
    d = cache_dir(youtube_id)
    # Keep: audio.wav, vocals.wav, lyrics.json, manifest.json
    # Remove: raw download, demucs work directory
    for pattern in ["audio_raw.*"]:
        for f in d.glob(pattern):
            f.unlink(missing_ok=True)
            logger.debug("Removed temp file: %s", f)

    demucs_dir = d / "demucs"
    if demucs_dir.exists():
        shutil.rmtree(demucs_dir)
        logger.debug("Removed demucs work dir: %s", demucs_dir)


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1_048_576
