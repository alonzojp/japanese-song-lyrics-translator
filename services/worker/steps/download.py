"""
Step 1 — Download audio from YouTube via yt-dlp.
Output: {cache_dir}/{youtube_id}/audio.wav
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import yt_dlp

logger = logging.getLogger(__name__)


def download_audio(
    youtube_url: str,
    output_dir: Path,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> tuple[Path, dict]:
    """
    Download and convert audio to WAV.
    Returns (wav_path, info_dict).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / "audio.wav"

    if wav_path.exists():
        logger.info("Cache hit — skipping download for %s", output_dir.name)
        if progress_cb:
            progress_cb(100)
        return wav_path, {}

    def _hook(d: dict) -> None:
        if progress_cb and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 1
            downloaded = d.get("downloaded_bytes", 0)
            progress_cb(int(downloaded / total * 90))
        elif d.get("status") == "finished" and progress_cb:
            progress_cb(95)

    ydl_opts: dict = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "audio.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "progress_hooks": [_hook],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)

    if progress_cb:
        progress_cb(100)

    return wav_path, info or {}
