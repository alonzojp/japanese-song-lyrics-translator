"""
Forced acoustic alignment backend.

Activated when the quality gate detects catastrophic ASR failure
(repeated choruses, dense harmonies, or grossly collapsed Whisper output).

Strategy
--------
1. Load the WhisperX Japanese wav2vec2 CTC model (already cached on disk).
2. Run the full vocals waveform through the model → CTC log-probability matrix
   with shape [T_frames, vocab_size].
3. Convert the official lyrics to a flat character-token sequence using the
   model's own vocabulary.  Characters not in the vocabulary are skipped; the
   caller will see reduced word coverage but still gets correct line ordering.
4. Run torchaudio.functional.forced_align over the full sequence in one pass.
   Because CTC forced alignment is monotone it naturally handles repeated
   chorus occurrences without collapsing them — each occurrence is pinned to
   its own acoustic position.
5. Map token-level frame timestamps back to per-line segments in the same
   format that run_alignment_postprocess() expects.

Output schema (identical to force_align() in aligner.py)
---------------------------------------------------------
[{
    "text":  str,                       # original lyric line
    "start": float,                     # line start time (seconds)
    "end":   float,                     # line end time (seconds)
    "words": [{"word": str, "start": float, "end": float, "score": float}]
}]

Notes
-----
* Words are character-level (one entry per kana/kanji token) rather than
  word-level.  The downstream build_lyric_lines() and visual_timing code
  consume word arrays at the character level without issues.
* Lines with zero vocabulary coverage (all characters OOV) receive a gap-fill
  placement based on the preceding line's end time; these are flagged in logs.
* For audio longer than _CHUNK_DURATION_S the model is run in overlapping
  chunks to avoid OOM on long songs; emission matrices are stitched at frame
  boundaries.
"""
from __future__ import annotations

import logging
import unicodedata
from pathlib import Path
from typing import Optional

from config import WHISPER_LANGUAGE

logger = logging.getLogger(__name__)

# Sliding-window parameters for long audio
_CHUNK_DURATION_S  = 180.0    # process at most this many seconds per chunk
_CHUNK_OVERLAP_S   = 5.0      # overlap between consecutive chunks (seconds)

# Gap-fill duration for OOV lines (no vocabulary coverage)
_OOV_FILL_S        = 2.0

# Minimum number of tokens for a line to be considered "covered"
_MIN_LINE_TOKENS   = 1


def run_forced_alignment(
    vocals_path:        Path,
    official_line_texts: list[str],
    job_log=None,
) -> list[dict]:
    """
    Align official lyrics directly to audio using wav2vec2 CTC forced alignment.

    Parameters
    ----------
    vocals_path         : path to the pre-processed vocals WAV (16 kHz mono)
    official_line_texts : plain text lyric lines in display order
    job_log             : optional JobLogger instance

    Returns
    -------
    Segments list compatible with run_alignment_postprocess().
    Raises RuntimeError if the model cannot be loaded or forced_align fails.
    """
    import torch
    import torch.nn.functional as F
    import torchaudio
    import whisperx

    def _log(msg: str) -> None:
        if job_log:
            job_log.info(msg, stage="align")
        logger.info(msg)

    def _warn(msg: str) -> None:
        if job_log:
            job_log.warning(msg, stage="align")
        logger.warning(msg)

    device = _get_device()
    _log(
        f"[forced_align] Starting acoustic forced alignment — "
        f"{len(official_line_texts)} lines, device={device}"
    )

    # ── 1. Load alignment model ────────────────────────────────────────────────
    _log("[forced_align] Loading Japanese wav2vec2 alignment model …")
    try:
        align_model, align_meta = whisperx.load_align_model(
            language_code=WHISPER_LANGUAGE,
            device=device,
        )
    except Exception as exc:
        raise RuntimeError(
            f"[forced_align] Could not load alignment model: {exc}"
        ) from exc

    dictionary: dict[str, int] = align_meta.get("dictionary", {})
    if not dictionary:
        raise RuntimeError(
            "[forced_align] align_meta['dictionary'] is empty — "
            "cannot map lyrics to token IDs"
        )

    # The CTC blank token is the padding token at index 0 in most wav2vec2 models.
    # WhisperX does not expose blank_id directly; we derive it as the lowest
    # token ID not present in the character dictionary.
    char_ids = set(dictionary.values())
    blank_id = 0
    while blank_id in char_ids:
        blank_id += 1

    _log(
        f"[forced_align] Vocabulary: {len(dictionary)} chars, blank_id={blank_id}"
    )

    # ── 2. Load audio ──────────────────────────────────────────────────────────
    _log(f"[forced_align] Loading audio: {vocals_path.name}")
    audio_np = whisperx.load_audio(str(vocals_path))   # float32 numpy, 16 kHz
    audio_duration = len(audio_np) / 16_000
    _log(f"[forced_align] Audio duration: {audio_duration:.1f}s")

    import torch
    audio_tensor = torch.tensor(audio_np, dtype=torch.float32)

    # ── 3. Emission matrix (chunked for long songs) ────────────────────────────
    log_probs = _get_emission_matrix(
        align_model, audio_tensor, audio_duration, device, _log
    )
    # log_probs: [T_frames, C]

    n_frames   = log_probs.shape[0]
    frame_rate = n_frames / audio_duration
    _log(
        f"[forced_align] Emission matrix: {n_frames} frames, "
        f"frame_rate={frame_rate:.2f} Hz"
    )

    # ── 4. Build token sequence from lyrics ────────────────────────────────────
    all_tokens:       list[int]            = []
    line_token_ranges: list[tuple[int, int]] = []  # (inclusive_start, exclusive_end)

    for text in official_line_texts:
        normalized = unicodedata.normalize("NFKC", text)
        t_start = len(all_tokens)
        for ch in normalized:
            ch_n = unicodedata.normalize("NFKC", ch)
            if ch_n in dictionary:
                all_tokens.append(dictionary[ch_n])
            elif ch_n.lower() in dictionary:
                all_tokens.append(dictionary[ch_n.lower()])
        line_token_ranges.append((t_start, len(all_tokens)))

    if not all_tokens:
        raise RuntimeError(
            "[forced_align] No characters in official lyrics matched the model "
            "vocabulary.  Verify that official_line_texts contain kana/kanji."
        )

    total_chars = sum(
        len(unicodedata.normalize("NFKC", t)) for t in official_line_texts
    )
    vocab_coverage = len(all_tokens) / max(total_chars, 1)
    _log(
        f"[forced_align] Token sequence: {len(all_tokens)} tokens "
        f"from {total_chars} input chars (vocab coverage={vocab_coverage:.1%})"
    )
    if vocab_coverage < 0.30:
        _warn(
            f"[forced_align] Low vocabulary coverage ({vocab_coverage:.1%}) — "
            "many kanji may be OOV; consider adding pykakasi for kanji→kana conversion"
        )

    # CTC constraint: len(tokens) ≤ (n_frames + 1) // 2
    if len(all_tokens) > (n_frames + 1) // 2:
        raise RuntimeError(
            f"[forced_align] Token sequence ({len(all_tokens)}) exceeds CTC limit "
            f"({(n_frames + 1) // 2}) for {n_frames} audio frames. "
            "Lyrics may be longer than the audio."
        )

    # ── 5. CTC forced alignment ────────────────────────────────────────────────
    _log("[forced_align] Running torchaudio.functional.forced_align …")

    targets        = torch.tensor(all_tokens, dtype=torch.int32).unsqueeze(0)   # [1, N]
    input_lengths  = torch.tensor([n_frames],          dtype=torch.int64)       # [1]
    target_lengths = torch.tensor([len(all_tokens)],   dtype=torch.int64)       # [1]
    lp_batch       = log_probs.unsqueeze(0)                                     # [1, T, C]

    try:
        token_frames, token_scores = torchaudio.functional.forced_align(
            lp_batch,
            targets,
            input_lengths,
            target_lengths,
            blank=blank_id,
        )
    except AttributeError as exc:
        raise RuntimeError(
            f"[forced_align] torchaudio.functional.forced_align not available "
            f"(torchaudio may be too old): {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"[forced_align] forced_align failed: {exc}"
        ) from exc

    # [N] tensors, one entry per token
    tok_times  = token_frames[0].float() / frame_rate   # seconds
    tok_scores = token_scores[0].float()                 # 0–1

    _log(
        f"[forced_align] Alignment complete — "
        f"first={float(tok_times[0]):.3f}s  last={float(tok_times[-1]):.3f}s"
    )

    # ── 6. Map token timestamps → line segments ────────────────────────────────
    del align_model
    _free_vram()

    segments: list[dict] = []
    oov_lines = 0

    for line_idx, (text, (t_start, t_end)) in enumerate(
        zip(official_line_texts, line_token_ranges)
    ):
        if t_end - t_start < _MIN_LINE_TOKENS:
            # OOV line — gap fill after the previous line
            prev_end = segments[-1]["end"] if segments else 0.0
            seg = {
                "text":  text,
                "start": round(prev_end, 3),
                "end":   round(prev_end + _OOV_FILL_S, 3),
                "words": [],
            }
            segments.append(seg)
            oov_lines += 1
            _warn(
                f"[forced_align] L{line_idx:02d}: OOV gap-fill "
                f"{prev_end:.3f}–{prev_end + _OOV_FILL_S:.3f}s  '{text[:40]}'"
            )
            continue

        line_times  = tok_times[t_start:t_end]
        line_scores = tok_scores[t_start:t_end]

        seg_start = float(line_times[0])

        # Estimate end: last token start + duration of one token (using gap to
        # next line's first token when available, else 1 frame).
        if t_end < len(all_tokens):
            next_start    = float(tok_times[t_end])
            token_dur_est = (next_start - seg_start) / max(t_end - t_start, 1)
            seg_end       = min(
                float(line_times[-1]) + token_dur_est,
                next_start,
            )
        else:
            seg_end = float(line_times[-1]) + (5.0 / frame_rate)

        # Character-level word entries
        normalized = unicodedata.normalize("NFKC", text)
        chars = [
            ch for ch in normalized
            if unicodedata.normalize("NFKC", ch) in dictionary
            or unicodedata.normalize("NFKC", ch).lower() in dictionary
        ]
        words: list[dict] = []
        for j, (ch, t, sc) in enumerate(zip(chars, line_times, line_scores)):
            w_end = float(line_times[j + 1]) if j + 1 < len(line_times) else seg_end
            words.append({
                "word":  ch,
                "start": round(float(t),  3),
                "end":   round(w_end,     3),
                "score": round(float(sc), 3),
            })

        avg_sc = float(line_scores.mean())
        seg = {
            "text":  text,
            "start": round(seg_start, 3),
            "end":   round(seg_end,   3),
            "words": words,
        }
        segments.append(seg)
        _log(
            f"[forced_align] L{line_idx:02d}: {seg_start:.3f}–{seg_end:.3f}s "
            f"({t_end-t_start} tokens, avg_score={avg_sc:.3f})  '{text[:40]}'"
        )

    if oov_lines:
        _warn(
            f"[forced_align] {oov_lines}/{len(official_line_texts)} lines were OOV "
            "(gap-filled); alignment quality for those lines will be low)"
        )

    return segments


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_emission_matrix(
    align_model,
    audio_tensor,
    audio_duration: float,
    device: str,
    log_fn,
) -> "torch.Tensor":
    """
    Run the full waveform through the wav2vec2 model and return log_probs [T, C].
    Processes in chunks when the audio is longer than _CHUNK_DURATION_S to
    avoid OOM on transformer attention.
    """
    import torch
    import torch.nn.functional as F

    sr        = 16_000
    chunk_len = int(_CHUNK_DURATION_S * sr)
    overlap   = int(_CHUNK_OVERLAP_S  * sr)

    n_samples = audio_tensor.shape[0]

    if n_samples <= chunk_len:
        # Short enough — process in one pass
        normed = F.layer_norm(audio_tensor, audio_tensor.shape)
        with torch.no_grad():
            out      = align_model(normed.unsqueeze(0).to(device))
            lp_chunk = F.log_softmax(out.logits[0], dim=-1).cpu()
        return lp_chunk   # [T, C]

    # Long audio — chunk with overlap and stitch
    log_fn(
        f"[forced_align] Audio >{_CHUNK_DURATION_S:.0f}s — "
        f"processing in {int(_CHUNK_DURATION_S)}s chunks …"
    )
    chunks: list["torch.Tensor"] = []
    pos = 0
    while pos < n_samples:
        end    = min(pos + chunk_len, n_samples)
        seg    = audio_tensor[pos:end]
        normed = F.layer_norm(seg, seg.shape)
        with torch.no_grad():
            out  = align_model(normed.unsqueeze(0).to(device))
            chunk_lp = F.log_softmax(out.logits[0], dim=-1).cpu()

        # Drop the overlap region from all but the first chunk
        if chunks:
            # frames corresponding to _CHUNK_OVERLAP_S — trim leading overlap
            total_frames = chunk_lp.shape[0]
            dur_chunk    = (end - pos) / sr
            overlap_fr   = int(total_frames * _CHUNK_OVERLAP_S / dur_chunk)
            chunk_lp     = chunk_lp[overlap_fr:]

        chunks.append(chunk_lp)
        pos += chunk_len - overlap

    return torch.cat(chunks, dim=0)   # [T_total, C]


def _get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _free_vram() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
