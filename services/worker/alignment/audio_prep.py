"""
Audio preprocessing for ASR (Automatic Speech Recognition).

Takes vocals.wav (44.1 kHz stereo, Demucs output) and produces
vocals_asr.wav (16 kHz mono, filtered, normalized) optimised for Whisper.

ffmpeg chain:
  highpass=f=80          remove sub-80Hz (kick drum bleed, low rumble)
  lowpass=f=8000         speech intelligibility cap
  anlmdn=s=0.00005       non-local means denoiser (gentle pass)
  loudnorm=I=-16:TP=-1   normalize to -16 LUFS (Whisper sweet-spot)
  16 kHz mono            Whisper's native sample rate
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

ASR_FILE = "vocals_asr.wav"


def prepare_audio_for_asr(
    input_path: Path,
    output_dir: Path,
    progress_cb: Optional[Callable[[int], None]] = None,
    job_log=None,
) -> Path:
    """
    Apply speech-optimised ffmpeg pipeline to input_path.
    Returns path to the processed ASR audio file.
    Skips if output already exists.
    """
    out = output_dir / ASR_FILE

    def log(msg: str) -> None:
        if job_log:
            job_log.info(msg, stage="transcribe")
        logger.info(msg)

    if out.exists():
        log(f"ASR audio already prepared: {out.name}")
        if progress_cb:
            progress_cb(100)
        return out

    log(f"Preparing audio for ASR: {input_path.name} → {out.name}")

    af_chain = (
        "highpass=f=80,"
        "lowpass=f=8000,"
        "anlmdn=s=0.00005,"
        "loudnorm=I=-16:TP=-1:LRA=7:print_format=none"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-af", af_chain,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(out),
    ]

    if progress_cb:
        progress_cb(30)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg ASR prep failed (code {result.returncode}):\n{result.stderr[-1500:]}"
        )

    size_mb = out.stat().st_size / 1_048_576
    log(f"ASR audio ready: {size_mb:.1f} MB at 16 kHz mono")

    if progress_cb:
        progress_cb(100)
    return out
