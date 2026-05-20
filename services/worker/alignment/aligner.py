"""
Forced alignment — attaches precise word timestamps to transcript segments.

Priority:
  1. WhisperX native alignment  (best for Japanese, CTC-based)
  2. aeneas                     (DTW-based, language-configurable, good fallback)
  3. Gentle                     (local HTTP server, English-focused, last resort)
  4. Even distribution          (synthetic — spreads words uniformly per segment)

Each method enriches existing segment dicts with per-word timing.
Returns (enriched_segments, method_used, per_word_scores).
"""
from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Callable, Optional

from config import WHISPER_LANGUAGE

logger = logging.getLogger(__name__)

AlignedSegments  = list[dict]   # same shape as transcriber output
AlignmentMethod  = str          # whisperx | aeneas | gentle | even


# ── WhisperX forced alignment ──────────────────────────────────────────────────

def _align_whisperx(
    segments:   AlignedSegments,
    audio_path: Path,
    job_log,
) -> tuple[AlignedSegments, AlignmentMethod]:
    import whisperx

    device = _get_device()
    if job_log:
        job_log.info(f"[align] WhisperX forced alignment on {audio_path.name}", stage="transcribe")

    audio        = whisperx.load_audio(str(audio_path))
    align_model, align_meta = whisperx.load_align_model(
        language_code=WHISPER_LANGUAGE,
        device=device,
    )
    aligned = whisperx.align(
        segments, align_model, align_meta, audio, device,
        return_char_alignments=False,
    )

    # Free VRAM
    del align_model
    _free_vram()

    return aligned.get("segments", segments), "whisperx"


# ── aeneas forced alignment ────────────────────────────────────────────────────

def _align_aeneas(
    segments:   AlignedSegments,
    audio_path: Path,
    job_log,
) -> tuple[AlignedSegments, AlignmentMethod]:
    """
    Uses aeneas to produce line-level timestamps from text + audio.
    Word-level timing is then filled with even distribution within each line.

    aeneas is DTW-based and language-configurable — works well for Japanese
    when configured with the appropriate TTS voice.
    """
    from aeneas.executetask import ExecuteTask
    from aeneas.task import Task

    if job_log:
        job_log.info("[align] aeneas forced alignment", stage="transcribe")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path   = Path(tmp)
        text_path  = tmp_path / "text.txt"
        sync_path  = tmp_path / "sync.json"

        # Write one line of text per segment
        lines = [seg.get("text", "").strip() for seg in segments]
        text_path.write_text("\n".join(lines), encoding="utf-8")

        config = (
            f"task_language=ja"
            f"|is_audio_file_detect_head_max=0.00"
            f"|is_audio_file_detect_tail_max=0.00"
            f"|os_task_file_format=json"
            f"|os_task_file_head_tail_format=hidden"
        )
        task = Task(config_string=config)
        task.audio_file_path_absolute = str(audio_path.resolve())
        task.text_file_path_absolute  = str(text_path.resolve())
        task.sync_map_file_path_absolute = str(sync_path.resolve())

        ExecuteTask(task).execute()
        task.output_sync_map_file()

        import json
        sync_data = json.loads(sync_path.read_text(encoding="utf-8"))

    # aeneas sync map: {fragments: [{id, begin, end, lines: [...]}]}
    fragments = sync_data.get("fragments", [])
    enriched  = list(segments)

    for i, (seg, frag) in enumerate(zip(enriched, fragments)):
        seg_start = float(frag.get("begin", seg.get("start", 0)))
        seg_end   = float(frag.get("end",   seg.get("end",   seg_start + 1)))
        seg["start"] = seg_start
        seg["end"]   = seg_end
        # Fill words with even distribution (aeneas is line-level)
        if not seg.get("words"):
            words_text = re.findall(r"\S+", seg.get("text", ""))
            seg["words"] = _distribute_evenly(words_text, seg_start, seg_end, score=0.3)

    return enriched, "aeneas"


# ── Gentle forced alignment ────────────────────────────────────────────────────

def _align_gentle(
    segments:   AlignedSegments,
    audio_path: Path,
    job_log,
) -> tuple[AlignedSegments, AlignmentMethod]:
    """
    Calls a locally running Gentle server (default: http://localhost:8765).
    Gentle is English-focused — for Japanese this is a weak alignment.
    """
    import httpx

    gentle_url = "http://localhost:8765/transcriptions"
    transcript = " ".join(seg.get("text", "") for seg in segments)

    if job_log:
        job_log.info("[align] Gentle forced alignment (local server)", stage="transcribe")

    with open(audio_path, "rb") as f:
        resp = httpx.post(
            gentle_url,
            data={"transcript": transcript},
            files={"audio": (audio_path.name, f, "audio/wav")},
            params={"async": "false"},
            timeout=60.0,
        )
    resp.raise_for_status()
    data = resp.json()

    # Map Gentle word alignments back to segments
    gentle_words = [
        w for w in data.get("words", [])
        if w.get("case") == "success"
    ]

    enriched = _back_project_words_to_segments(segments, gentle_words, score=0.5)
    return enriched, "gentle"


# ── Even distribution fallback ─────────────────────────────────────────────────

def _align_even(
    segments: AlignedSegments,
    job_log,
) -> tuple[AlignedSegments, AlignmentMethod]:
    """
    Distribute words evenly across each segment's time span.
    Score = 0.0 to signal synthetic (not real) alignment.
    """
    if job_log:
        job_log.info(
            "[align] Falling back to even word distribution (no forced aligner available)",
            stage="transcribe",
        )
    enriched = list(segments)
    for seg in enriched:
        if not seg.get("words"):
            words_text = re.findall(r"\S+", seg.get("text", ""))
            seg["words"] = _distribute_evenly(
                words_text,
                seg.get("start", 0.0),
                seg.get("end",   0.0),
                score=0.0,
            )
    return enriched, "even"


# ── Public entry point ─────────────────────────────────────────────────────────

def force_align(
    segments:   AlignedSegments,
    audio_path: Path,
    progress_cb: Optional[Callable[[int], None]] = None,
    job_log=None,
) -> tuple[AlignedSegments, AlignmentMethod]:
    """
    Attach word-level timestamps to segments using the best available aligner.
    Returns (enriched_segments, method_name).
    """
    # If the transcriber already produced word-level timing (e.g. WhisperX or
    # faster-whisper with word_timestamps=True), still run WhisperX alignment
    # for improved accuracy — but skip if words already have good scores.
    already_aligned = all(
        seg.get("words") and
        all(w.get("score", 0) > 0.3 for w in seg["words"])
        for seg in segments
        if seg.get("text", "").strip()
    )

    if already_aligned:
        if job_log:
            job_log.info("[align] Transcriber already provided word timestamps — skipping forced alignment", stage="transcribe")
        if progress_cb:
            progress_cb(100)
        return segments, "transcriber"

    aligners = [
        ("whisperx", lambda: _align_whisperx(segments, audio_path, job_log)),
        ("aeneas",   lambda: _align_aeneas(segments, audio_path, job_log)),
        ("gentle",   lambda: _align_gentle(segments, audio_path, job_log)),
    ]

    for name, fn in aligners:
        try:
            result, method = fn()
            if progress_cb:
                progress_cb(100)
            return result, method
        except ImportError:
            if job_log:
                job_log.info(f"[align] {name} not available, trying next", stage="transcribe")
        except Exception as exc:
            if job_log:
                job_log.warning(f"[align] {name} failed: {exc}", stage="transcribe")

    # Last resort
    result, method = _align_even(segments, job_log)
    if progress_cb:
        progress_cb(100)
    return result, method


# ── Helpers ────────────────────────────────────────────────────────────────────

def _distribute_evenly(
    words: list[str],
    start: float,
    end:   float,
    score: float = 0.0,
) -> list[dict]:
    if not words:
        return []
    duration = max(0.0, end - start)
    per_word = duration / len(words)
    return [
        {
            "word":  w,
            "start": round(start + i * per_word, 3),
            "end":   round(start + (i + 1) * per_word, 3),
            "score": score,
        }
        for i, w in enumerate(words)
    ]


def _back_project_words_to_segments(
    segments:     AlignedSegments,
    gentle_words: list[dict],
    score:        float,
) -> AlignedSegments:
    """
    For each segment, collect Gentle words that fall within [start, end].
    Words not matched → fill remaining with even distribution.
    """
    enriched = list(segments)
    for seg in enriched:
        seg_start = seg.get("start", 0.0)
        seg_end   = seg.get("end",   0.0)
        matched   = [
            {
                "word":  w.get("word", ""),
                "start": w.get("startOffset", seg_start) / 1000,
                "end":   w.get("endOffset",   seg_end)   / 1000,
                "score": score,
            }
            for w in gentle_words
            if seg_start <= (w.get("startOffset", 0) / 1000) <= seg_end
        ]
        if not matched:
            words_text = re.findall(r"\S+", seg.get("text", ""))
            matched    = _distribute_evenly(words_text, seg_start, seg_end, score=score)
        seg["words"] = matched
    return enriched


def _get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _free_vram() -> None:
    try:
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
