"""
Step 2 — Vocal separation via Demucs (HTDemucs).
Input:  audio.wav (full mix)
Output: vocals.wav
Skippable via SKIP_VOCAL_SEPARATION=true (falls back to full mix).
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

from config import DEMUCS_MODEL, SKIP_VOCAL_SEPARATION

logger = logging.getLogger(__name__)


def separate_vocals(
    audio_path: Path,
    output_dir: Path,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> Path:
    """
    Run Demucs to isolate vocals.
    Returns path to vocals.wav (or original audio if separation is skipped).
    """
    if SKIP_VOCAL_SEPARATION:
        logger.info("Vocal separation skipped (SKIP_VOCAL_SEPARATION=true)")
        if progress_cb:
            progress_cb(100)
        return audio_path

    vocals_path = output_dir / "vocals.wav"
    if vocals_path.exists():
        logger.info("Cache hit — skipping separation for %s", output_dir.name)
        if progress_cb:
            progress_cb(100)
        return vocals_path

    if progress_cb:
        progress_cb(5)

    # Demucs writes to: {output_dir}/demucs/{model}/{stem}/{audio_stem}.wav
    demucs_out = output_dir / "demucs"
    cmd = [
        "python", "-m", "demucs",
        "--two-stems", "vocals",
        "--model", DEMUCS_MODEL,
        "--out", str(demucs_out),
        str(audio_path),
    ]

    logger.info("Running Demucs: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed:\n{result.stderr}")

    if progress_cb:
        progress_cb(90)

    # Find the generated vocals file and copy it to a flat path
    stem = audio_path.stem
    candidate = demucs_out / DEMUCS_MODEL / stem / "vocals.wav"
    if not candidate.exists():
        # Fallback: search recursively
        matches = list(demucs_out.rglob("vocals.wav"))
        if not matches:
            raise FileNotFoundError("Demucs did not produce vocals.wav")
        candidate = matches[0]

    shutil.copy2(candidate, vocals_path)

    if progress_cb:
        progress_cb(100)

    return vocals_path
