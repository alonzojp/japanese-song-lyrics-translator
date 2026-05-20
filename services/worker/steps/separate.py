"""
Step 2 — Vocal separation via Demucs (HTDemucs).
Input:  audio.wav (full mix, 44.1 kHz stereo)
Output: vocals.wav

Cache behaviour: skips entirely if manifest marks separate as complete.
Skippable via SKIP_VOCAL_SEPARATION=true (passes audio.wav straight to Whisper).
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

from cache import is_stage_complete, mark_stage_complete
from config import DEMUCS_MODEL, SKIP_VOCAL_SEPARATION

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[int], None]]


def separate_vocals(
    audio_path:  Path,
    output_dir:  Path,
    youtube_id:  str,
    progress_cb: ProgressCallback = None,
    job_log=None,
) -> Path:
    """
    Run Demucs to isolate vocals.
    Returns path to vocals.wav, or original audio_path if separation is skipped.
    """
    def log(msg: str) -> None:
        if job_log:
            job_log.info(msg, stage="separate")
        logger.info(msg)

    vocals_path = output_dir / "vocals.wav"

    if SKIP_VOCAL_SEPARATION:
        log("Vocal separation skipped (SKIP_VOCAL_SEPARATION=true)")
        if progress_cb:
            progress_cb(100)
        return audio_path

    if is_stage_complete(youtube_id, "separate") and vocals_path.exists():
        log(f"Cache hit — skipping separation for {youtube_id}")
        if progress_cb:
            progress_cb(100)
        return vocals_path

    log(f"Separating vocals with Demucs model={DEMUCS_MODEL} …")
    if progress_cb:
        progress_cb(5)

    demucs_out = output_dir / "demucs"
    cmd = [
        "python", "-m", "demucs",
        "--two-stems", "vocals",
        "--model",     DEMUCS_MODEL,
        "--out",       str(demucs_out),
        str(audio_path),
    ]

    logger.debug("Demucs: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed:\n{result.stderr[-2000:]}")

    if progress_cb:
        progress_cb(88)

    # Demucs writes: {demucs_out}/{model}/{stem}/vocals.wav
    stem = audio_path.stem
    candidate = demucs_out / DEMUCS_MODEL / stem / "vocals.wav"
    if not candidate.exists():
        matches = list(demucs_out.rglob("vocals.wav"))
        if not matches:
            raise FileNotFoundError("Demucs did not produce vocals.wav")
        candidate = matches[0]

    shutil.copy2(candidate, vocals_path)

    mark_stage_complete(youtube_id, "separate", {"vocalsPath": "vocals.wav"})
    log("Vocal separation complete → vocals.wav")

    if progress_cb:
        progress_cb(100)
    return vocals_path
