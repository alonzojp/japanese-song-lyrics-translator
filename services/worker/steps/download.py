"""
Step 1 — Download + normalize audio.

Sub-steps:
  a) yt-dlp   → audio_raw.{ext}   (best-quality original)
  b) ffmpeg   → audio.wav         (44.1 kHz stereo, EBU R128 loudnorm, optional silence trim)

Cache behaviour:
  - If audio.wav already exists in the manifest, skip everything.
  - If audio_raw exists but audio.wav does not, re-run only the ffmpeg step.

Returns:
  (audio_wav_path, metadata_dict)
  metadata keys: title, uploader, duration, thumbnail_url, webpage_url
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Callable, Optional

import yt_dlp

from cache import get_stage_data, is_stage_complete, mark_stage_complete
from config import (
    AUDIO_LUFS_TARGET,
    AUDIO_SAMPLE_RATE,
    AUDIO_TRUE_PEAK,
    SILENCE_DURATION,
    SILENCE_THRESHOLD,
    TRIM_SILENCE,
)

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[int], None]]


# ── yt-dlp download ────────────────────────────────────────────────────────────

def _ydl_opts(output_dir: Path, progress_cb: ProgressCallback) -> dict:
    def _hook(d: dict) -> None:
        if progress_cb and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 1
            done  = d.get("downloaded_bytes", 0)
            progress_cb(int(done / total * 45))  # 0–45 %
        elif d.get("status") == "finished" and progress_cb:
            progress_cb(50)

    return {
        # Best audio; prefer opus/m4a/webm (smaller than wav)
        "format":           "bestaudio/best",
        "outtmpl":          str(output_dir / "audio_raw.%(ext)s"),
        "progress_hooks":   [_hook],
        "quiet":            True,
        "no_warnings":      True,
        "extract_flat":     False,
        # Write thumbnail so we have it locally
        "writethumbnail":   False,
        # No playlist support — single video only
        "noplaylist":       True,
    }


def _extract_metadata(info: dict) -> dict:
    duration = info.get("duration") or 0
    return {
        "title":         info.get("title")    or info.get("fulltitle") or "",
        "uploader":      info.get("uploader") or info.get("channel")  or "",
        "duration":      float(duration),
        "thumbnailUrl":  info.get("thumbnail") or "",
        "webpageUrl":    info.get("webpage_url") or "",
    }


# ── ffmpeg normalisation ───────────────────────────────────────────────────────

def _build_af(trim: bool) -> str:
    """Build the ffmpeg -af filter string."""
    loudnorm = (
        f"loudnorm=I={AUDIO_LUFS_TARGET}:TP={AUDIO_TRUE_PEAK}:LRA=11"
        ":print_format=none"
    )
    if not trim:
        return loudnorm

    # Two-pass silence removal: strip leading and trailing silence
    sr_start = (
        f"silenceremove="
        f"start_periods=1:start_silence={SILENCE_DURATION}:start_threshold={SILENCE_THRESHOLD}"
    )
    sr_end = (
        f"silenceremove="
        f"stop_periods=-1:stop_silence={SILENCE_DURATION}:stop_threshold={SILENCE_THRESHOLD}"
        ":detection=peak"
    )
    return f"{sr_start},{loudnorm},{sr_end}"


def _run_ffmpeg(raw_path: Path, out_path: Path, progress_cb: ProgressCallback) -> None:
    if progress_cb:
        progress_cb(55)

    af = _build_af(TRIM_SILENCE)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(raw_path),
        "-af", af,
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", "2",             # stereo — Demucs works best with stereo
        "-c:a", "pcm_s16le",    # uncompressed WAV
        str(out_path),
    ]

    logger.debug("ffmpeg: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # ffmpeg writes to stderr even on success; only raise if exit code non-zero
        raise RuntimeError(
            f"ffmpeg failed (code {result.returncode}):\n{result.stderr[-2000:]}"
        )

    if progress_cb:
        progress_cb(95)


# ── Public entry point ─────────────────────────────────────────────────────────

def download_audio(
    youtube_url:  str,
    youtube_id:   str,
    output_dir:   Path,
    progress_cb:  ProgressCallback = None,
    job_log=None,
) -> tuple[Path, dict]:
    """
    Download and normalise audio for a YouTube video.
    Returns (audio_wav_path, metadata_dict).

    Cache hits:
    - If the download stage is already complete (manifest + file present), returns
      immediately with the stored metadata.
    - If audio_raw exists but audio.wav does not, skips yt-dlp and re-runs ffmpeg.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / "audio.wav"

    def log(msg: str) -> None:
        if job_log:
            job_log.info(msg, stage="download")
        logger.info(msg)

    # ── Cache hit: both stages done ────────────────────────────────────────────
    if is_stage_complete(youtube_id, "download") and wav_path.exists():
        meta = get_stage_data(youtube_id, "download").get("metadata", {})
        log(f"Cache hit — skipping download for {youtube_id}")
        if progress_cb:
            progress_cb(100)
        return wav_path, meta

    # ── Step a: yt-dlp download ────────────────────────────────────────────────
    raw_path = _find_raw(output_dir)

    if raw_path is None:
        log(f"Downloading audio for {youtube_id} …")
        opts = _ydl_opts(output_dir, progress_cb)
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True) or {}
        raw_path = _find_raw(output_dir)
        if raw_path is None:
            raise FileNotFoundError("yt-dlp did not produce an audio file")
        metadata = _extract_metadata(info)
        log(
            f"Downloaded: {metadata.get('title', youtube_id)} "
            f"({metadata.get('duration', 0):.0f}s)"
        )
    else:
        # Raw file exists but wav doesn't — probably a partial previous run
        log(f"Raw audio found, re-running ffmpeg for {youtube_id}")
        metadata = get_stage_data(youtube_id, "download").get("metadata", {})

    # ── Step b: ffmpeg normalise ───────────────────────────────────────────────
    log(
        f"Normalising audio → {AUDIO_SAMPLE_RATE} Hz stereo, "
        f"{AUDIO_LUFS_TARGET} LUFS"
        + (" (silence trim enabled)" if TRIM_SILENCE else "")
    )
    _run_ffmpeg(raw_path, wav_path, progress_cb)

    file_size = wav_path.stat().st_size

    # ── Update manifest ────────────────────────────────────────────────────────
    mark_stage_complete(youtube_id, "download", {
        "audioPath":    "audio.wav",
        "rawPath":      raw_path.name,
        "fileSizeBytes": file_size,
        "metadata":     metadata,
    })

    log(f"Audio ready: {file_size / 1_048_576:.1f} MB → audio.wav")
    if progress_cb:
        progress_cb(100)

    return wav_path, metadata


def _find_raw(output_dir: Path) -> Optional[Path]:
    """Find the raw yt-dlp download (any audio extension)."""
    for ext in ("webm", "opus", "m4a", "mp4", "ogg", "mp3", "wav"):
        p = output_dir / f"audio_raw.{ext}"
        if p.exists():
            return p
    return None
