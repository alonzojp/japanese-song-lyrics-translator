"""
Job pipeline — orchestrates the four processing steps.

Cache-aware: each step checks its manifest entry before running.
Re-submitting the same YouTube video is instant when all stages are cached.

Stage weights (overall progress %):
  download    0–20
  separate   20–45
  transcribe 45–85
  align      85–100
"""
from __future__ import annotations

import logging
from pathlib import Path

from cache import (
    all_stages_complete,
    cache_dir,
    cleanup_temp_files,
    get_metadata,
    invalidate_stage,
    is_stage_complete,
    store_metadata,
)
from config import CACHE_DIR, SKIP_VOCAL_SEPARATION, WHISPER_MODEL
from logs import JobLogger
from models import JobStatus
from job_queue import get_job, set_job_status, update_job_step, update_step_message
import asyncio

from alignment.matcher       import run_offline_match
from alignment.selector      import select_best_alignment
from alignment.visual_timing import visual_timing_normalization, finalize_timeline, classify_gap, _INST_GAP_THRESH_S, _VOCAL_GAP_THRESH_S
from alignment.outputs       import save_canonical_lines, load_canonical_lines
from correction import correct_segments
from lyrics.fetcher import fetch_lyrics, get_cached as get_cached_lyrics
from lyrics.types import VideoInfo
from steps.download import download_audio
from steps.separate import separate_vocals
from steps.transcribe import run_transcription
from steps.align import run_alignment_postprocess

logger = logging.getLogger(__name__)

_WEIGHTS = {
    "download":   (0,  20),
    "separate":   (20, 45),
    "transcribe": (45, 85),
    "align":      (85, 100),
}


def _overall(step: str, pct: int) -> int:
    lo, hi = _WEIGHTS[step]
    return lo + int((hi - lo) * pct / 100)


def _msg_cb(job_id: str, step: str):
    """Returns a callback that updates the step's human-readable message."""
    def cb(message: str) -> None:
        update_step_message(job_id, step, message)
    return cb


def _timeline_grade(lines: list[dict]) -> tuple[int, str]:
    """
    Score the canonical timeline using misalignment artifact signals only.

    The phrase model classifies every inter-line gap:
      CONT   ≤ 1.5s  within-phrase; tight flow expected here.
      SOFT   1.5–4s  verse/chorus transition; structural, not an error.
      STRONG > 4s    instrumental break; structural, not an error.

    Only CONT-gap quality and display-duration artifacts are penalised.
    SOFT and STRONG boundaries carry zero penalty regardless of count or size —
    penalising them would treat correct musical structure as noise.

    Penalty signals (all are misalignment artifacts, not musical choices):
      short_lines  < 1.5s  over-compression left a line too brief to read     −8 each
      overlaps     < 0     finalize_timeline invariant violated                −20 each
      large_cont   > 0.5s  within-phrase dead air visible to the viewer        −3 each
      avg_cont excess      broad within-phrase flow degradation                 −10/s
    """
    n = len(lines)
    if n == 0:
        return 0, 'F'

    gaps = []
    durs = []
    for i, ln in enumerate(lines):
        start = ln.get('startTime', 0)
        end   = ln.get('endTime',   0)
        durs.append(max(0.0, end - start))
        if i + 1 < n:
            gap = lines[i + 1].get('startTime', end) - end
            gaps.append(gap)

    # ── Misalignment artifact signals ─────────────────────────────────────────
    short_lines = sum(1 for d in durs if d < 1.5)
    overlaps    = sum(1 for g in gaps if g < -0.01)

    # CONT gaps only: SOFT and STRONG excluded (they are structural, not artifacts)
    cont_gaps  = [g for g in gaps if classify_gap(g) == 'CONT']
    large_cont = [g for g in cont_gaps if g > 0.5]   # visible dead air within a phrase
    avg_cont   = sum(cont_gaps) / len(cont_gaps) if cont_gaps else 0.0

    score = 100
    score -= short_lines     * 8    # unreadable line — over-compression artifact
    score -= overlaps        * 20   # timeline integrity failure
    score -= len(large_cont) * 3    # within-phrase dead air (>0.5s gap)
    score -= max(0.0, avg_cont - 0.3) * 10  # excess avg within-phrase gap
    score  = max(0, min(100, int(score)))

    grade = (
        'A' if score >= 90 else
        'B' if score >= 75 else
        'C' if score >= 55 else
        'D' if score >= 35 else 'F'
    )
    return score, grade


def _write_canonical(job_cache: Path, jl) -> None:
    """
    Select the best alignment, normalize if needed, write canonical_lines.json.

    Called exactly once per job, at pipeline completion.
    After this point the result API reads canonical_lines.json directly —
    no re-selection, no scoring, no mutation happens during GET requests.

    Acoustic lines are already normalized by run_alignment_postprocess().
    DTW lines bypass that step and are normalized here so both paths produce
    identical output structure (mora-compressed, gap-aware, finalized).
    """
    selection = select_best_alignment(job_cache, job_log=jl)
    lines  = selection["lines"]
    source = selection["source"]   # "acoustic" | "dtw"
    meta   = selection["selection_meta"]

    if not lines:
        jl.warning("[canonical] Selection returned no lines — canonical_lines.json not written")
        return

    jl.info(
        f"[canonical] source={source} "
        f"acoustic={selection['acoustic_score']:.3f} "
        f"dtw={selection['dtw_score']:.3f}: {meta.get('reason', '')}",
        stage="align",
    )

    # DTW lines have not passed through visual_timing_normalization / finalize_timeline.
    # Acoustic lines are already normalized — running them through again would
    # double-compress them, so we only normalize on the DTW path.
    if source == "dtw":
        visual_timing_normalization(lines, job_log=jl)
        finalize_timeline(lines, job_log=jl)
        jl.info(
            f"[canonical] DTW lines normalized through display pipeline ({len(lines)} lines)",
            stage="align",
        )

    # ── Repeat-gap fill (applies regardless of acoustic vs DTW selection) ────
    # Detects vocal-energy gaps Whisper skipped due to deduplication and runs a
    # targeted WhisperX forced-alignment pass to fill them.  No-op on songs
    # without repeated sections (energy check + coverage gate prevent false hits).
    try:
        from alignment.aligner import fill_canonical_repeat_gaps
        _yt_id = job_cache.name
        _lyrics = get_cached_lyrics(_yt_id)
        _official = _lyrics.get("lines", []) if _lyrics else []
        if _official:
            _asr   = job_cache / "vocals_asr.wav"
            _audio = _asr if _asr.exists() else job_cache / "audio.wav"
            if _audio.exists():
                lines = fill_canonical_repeat_gaps(lines, _official, _audio, jl)
    except Exception as _rg_exc:
        jl.warning(f"[repeat_gap] Canonical gap fill failed (non-fatal): {_rg_exc}",
                   stage="align")

    save_canonical_lines(job_cache, lines, source, meta)
    jl.info(f"[canonical] Wrote {len(lines)} lines → canonical_lines.json", stage="align")
    for i, ln in enumerate(lines):
        start = ln.get("startTime", 0)
        end   = ln.get("endTime", 0)
        dur   = round(end - start, 3)
        gap_str = ""
        if i + 1 < len(lines):
            gap = round(lines[i + 1].get("startTime", end) - end, 3)
            if gap < -0.01:
                boundary_tag = " ⚑OVERLAP"
            elif gap == 0.0:
                boundary_tag = " [CONT]"
            else:
                btype = classify_gap(gap)
                boundary_tag = "" if btype == 'CONT' else f" [{btype} {gap:.2f}s]"
            gap_str = f" gap={gap:.3f}s{boundary_tag}"
        jl.info(
            f"[canonical] L{i:02d}: {start:.3f}–{end:.3f}s "
            f"({dur:.3f}s){gap_str} '{ln.get('text', '')[:40]}'",
            stage="align",
        )

    # ── Timeline summary + quality grade ────────────────────────────────────
    score, grade = _timeline_grade(lines)
    n = len(lines)
    _gaps  = [lines[i+1].get("startTime", 0) - lines[i].get("endTime", 0) for i in range(n-1)]
    _durs  = [ln.get("endTime", 0) - ln.get("startTime", 0) for ln in lines]
    _avg_g = round(sum(_gaps) / len(_gaps), 3) if _gaps else 0.0
    _max_g = round(max(_gaps), 3) if _gaps else 0.0
    _med_d = round(sorted(_durs)[n // 2], 3) if _durs else 0.0
    _short = sum(1 for d in _durs if d < 1.5)
    _long  = sum(1 for d in _durs if d > 8.0)
    _n_strong = sum(1 for g in _gaps if classify_gap(g) == 'STRONG')
    _n_soft   = sum(1 for g in _gaps if classify_gap(g) == 'SOFT')
    _n_cont   = sum(1 for g in _gaps if classify_gap(g) == 'CONT')
    _n_phrases = 1 + _n_strong + _n_soft
    jl.info(
        f"[timeline_summary] total_lines={n} phrases={_n_phrases} "
        f"boundaries: {_n_cont}xCONT {_n_soft}xSOFT {_n_strong}xSTRONG | "
        f"avg_gap={_avg_g}s max_gap={_max_g}s median_dur={_med_d}s "
        f"short_lines={_short} long_lines={_long} "
        f"selector={source} grade={grade}({score})",
        stage="align",
    )


def run_job(job_id: str) -> None:
    job = get_job(job_id)
    if not job:
        logger.error("Job %s not found", job_id)
        return

    jl        = JobLogger(job_id)
    yt_id     = job.youtube_id
    job_cache = cache_dir(yt_id)

    set_job_status(job_id, JobStatus.processing)
    jl.info(f"Starting job {job_id[:8]}… youtube_id={yt_id}")

    # Invalidate transcribe/align stages when aligned_words.json is stale.
    # This must happen before all_stages_complete() so a version bump in
    # ALIGNER_VERSION automatically triggers re-alignment on the next job
    # without requiring a manual manifest clear.
    from alignment.outputs import load_aligned_words, ALIGNER_VERSION
    _aw_path = job_cache / "aligned_words.json"
    if _aw_path.exists() and load_aligned_words(job_cache) is None:
        jl.info(
            f"Aligner version mismatch — invalidating transcribe/align cache "
            f"(new version={ALIGNER_VERSION})"
        )
        invalidate_stage(yt_id, "transcribe")
        invalidate_stage(yt_id, "align")

    if all_stages_complete(yt_id):
        if get_cached_lyrics(yt_id):
            jl.info("All stages cached — instant completion")
            _mark_all_complete(job_id)
            # For jobs cached before canonical_lines.json existed, generate it now.
            # This is a pure read + normalize — no ML, sub-second, non-fatal.
            if not load_canonical_lines(job_cache):
                try:
                    _write_canonical(job_cache, jl)
                except Exception as exc:
                    jl.warning(f"Canonical write on cache-hit failed (non-fatal): {exc}")
            set_job_status(job_id, JobStatus.completed,
                           result_path=str(job_cache / "lyrics.json"))
            return
        # Audio cached but lyrics missing — fetch lyrics without re-running audio
        jl.info("Audio cached — fetching lyrics")
        _mark_all_complete(job_id)
        try:
            cached_meta = get_metadata(yt_id)
            video_info  = VideoInfo(
                youtube_id   = yt_id,
                youtube_url  = job.youtube_url,
                title        = cached_meta.get("title"),
                uploader     = cached_meta.get("uploader"),
                description  = cached_meta.get("description"),
                duration     = cached_meta.get("duration"),
            )
            loop = asyncio.new_event_loop()
            try:
                lyric_result = loop.run_until_complete(fetch_lyrics(video_info, job_log=jl))
            finally:
                loop.close()
            if lyric_result.get("lines"):
                jl.info(
                    f"Fetched {len(lyric_result['lines'])} lyrics lines "
                    f"via {lyric_result.get('provider', 'unknown')}"
                )
            match_result = run_offline_match(job_cache, yt_id, job_log=jl)
            if match_result:
                jl.info(
                    f"Offline match: {len(match_result['lines'])} lines, "
                    f"avg_conf={match_result['stats'].get('avgConfidence', 0):.2f}",
                )
            _write_canonical(job_cache, jl)
        except Exception as lyr_exc:
            jl.warning(f"Lyrics fetch/canonical failed (non-fatal): {lyr_exc}")
        set_job_status(job_id, JobStatus.completed,
                       result_path=str(job_cache / "lyrics.json"))
        return

    try:
        # ── Step 1: Download ───────────────────────────────────────────────────
        def dl_cb(pct: int) -> None:
            update_job_step(job_id, "download", JobStatus.processing, pct, _overall("download", pct))

        cached_dl = is_stage_complete(yt_id, "download")
        update_job_step(job_id, "download",
                        JobStatus.completed if cached_dl else JobStatus.processing,
                        100 if cached_dl else 0, _WEIGHTS["download"][0])

        audio_path, metadata = download_audio(
            youtube_url=job.youtube_url,
            youtube_id=yt_id,
            output_dir=job_cache,
            progress_cb=dl_cb,
            message_cb=_msg_cb(job_id, "download"),
            job_log=jl,
        )
        if metadata:
            store_metadata(yt_id, metadata)
        update_job_step(job_id, "download", JobStatus.completed, 100, _WEIGHTS["download"][1])

        # ── Step 2: Separate vocals ────────────────────────────────────────────
        def sep_cb(pct: int) -> None:
            update_job_step(job_id, "separate", JobStatus.processing, pct, _overall("separate", pct))

        cached_sep = is_stage_complete(yt_id, "separate") or SKIP_VOCAL_SEPARATION
        update_job_step(job_id, "separate",
                        JobStatus.completed if cached_sep else JobStatus.processing,
                        100 if cached_sep else 0, _WEIGHTS["separate"][0])

        vocals_path = separate_vocals(
            audio_path=audio_path,
            output_dir=job_cache,
            youtube_id=yt_id,
            progress_cb=sep_cb,
            message_cb=_msg_cb(job_id, "separate"),
            job_log=jl,
        )
        vocals_only = not SKIP_VOCAL_SEPARATION
        update_job_step(job_id, "separate", JobStatus.completed, 100, _WEIGHTS["separate"][1])

        # ── Pre-fetch official lyrics before transcription ─────────────────────
        # Having lyrics before alignment lets us use the acoustic model on the
        # correct text instead of Whisper's garbled output — much better timing.
        pre_lyrics: list[dict] = []
        try:
            cached_meta = get_metadata(yt_id) or (metadata or {})
            video_info_pre = VideoInfo(
                youtube_id  = yt_id,
                youtube_url = job.youtube_url,
                title       = cached_meta.get("title"),
                uploader    = cached_meta.get("uploader"),
                description = cached_meta.get("description"),
                duration    = cached_meta.get("duration"),
            )
            pre_loop = asyncio.new_event_loop()
            try:
                pre_result = pre_loop.run_until_complete(fetch_lyrics(video_info_pre, job_log=jl))
            finally:
                pre_loop.close()
            pre_lyrics = pre_result.get("lines", []) if pre_result else []
            if pre_lyrics:
                jl.info(
                    f"Pre-fetched {len(pre_lyrics)} lyrics lines via "
                    f"{pre_result.get('provider', 'unknown')} — will use for alignment",
                )
        except Exception as pre_exc:
            jl.warning(f"Pre-fetch lyrics failed (non-fatal): {pre_exc}")

        # ── Step 3: Transcription + forced alignment ───────────────────────────
        def tx_cb(pct: int) -> None:
            update_job_step(job_id, "transcribe", JobStatus.processing, pct, _overall("transcribe", pct))

        cached_tx = is_stage_complete(yt_id, "transcribe")
        update_job_step(job_id, "transcribe",
                        JobStatus.completed if cached_tx else JobStatus.processing,
                        100 if cached_tx else 0, _WEIGHTS["transcribe"][0])

        aligned_segments, backend, align_method = run_transcription(
            audio_path=vocals_path,
            output_dir=job_cache,
            youtube_id=yt_id,
            official_lyrics=pre_lyrics or None,
            progress_cb=tx_cb,
            message_cb=_msg_cb(job_id, "transcribe"),
            job_log=jl,
        )

        # LLM correction — fix kanji/kana errors in Whisper output using Groq
        aligned_segments = correct_segments(aligned_segments, job_log=jl)

        # ── Quality gate: detect catastrophic ASR failure ──────────────────────
        # Fires before postprocess so we can swap in forced acoustic alignment
        # when Whisper collapsed repeated choruses or produced unusable output.
        # The gate is a no-op when official lyrics are not available.
        if pre_lyrics:
            try:
                from alignment.acoustic_gate import detect_catastrophic_failure
                from alignment.forced_aligner import run_forced_alignment

                gate = detect_catastrophic_failure(
                    segments=aligned_segments,
                    official_lines=pre_lyrics,
                    align_method=align_method,
                    job_log=jl,
                )
                if gate.triggered:
                    jl.info(
                        f"[quality_gate] Switching to forced acoustic alignment — "
                        f"{gate.reasons}",
                        stage="align",
                    )
                    official_line_texts = [
                        l["text"] for l in pre_lyrics
                        if l.get("text", "").strip()
                    ]
                    forced_segs = run_forced_alignment(
                        vocals_path=vocals_path,
                        official_line_texts=official_line_texts,
                        job_log=jl,
                    )
                    aligned_segments = forced_segs
                    align_method     = "whisperx_forced"
                    backend          = "forced_acoustic"
                    jl.info(
                        f"[quality_gate] Forced alignment produced "
                        f"{len(forced_segs)} segments → continuing pipeline",
                        stage="align",
                    )
            except Exception as gate_exc:
                jl.warning(
                    f"[quality_gate] Gate/forced-alignment failed (non-fatal, "
                    f"using WhisperX output): {gate_exc}",
                    stage="align",
                )

        update_job_step(job_id, "transcribe", JobStatus.completed, 100, _WEIGHTS["transcribe"][1])

        # ── Step 4: Post-process → LyricLine files ─────────────────────────────
        def al_cb(pct: int) -> None:
            update_job_step(job_id, "align", JobStatus.processing, pct, _overall("align", pct))

        cached_al = is_stage_complete(yt_id, "align")
        update_job_step(job_id, "align",
                        JobStatus.completed if cached_al else JobStatus.processing,
                        100 if cached_al else 0, _WEIGHTS["align"][0])

        _lines, confidence = run_alignment_postprocess(
            aligned_segments=aligned_segments,
            output_dir=job_cache,
            youtube_id=yt_id,
            audio_path=vocals_path,
            backend=backend,
            method=align_method,
            vocals_only=vocals_only,
            progress_cb=al_cb,
            message_cb=_msg_cb(job_id, "align"),
            job_log=jl,
        )
        update_job_step(job_id, "align", JobStatus.completed, 100, 100)

        cleanup_temp_files(yt_id)

        # ── Optional step 4b: auto-fetch official lyrics ──────────────────────
        if not get_cached_lyrics(yt_id):
            try:
                video_info = VideoInfo(
                    youtube_id=yt_id,
                    youtube_url=job.youtube_url,
                    title=metadata.get("title")       if metadata else None,
                    uploader=metadata.get("uploader") if metadata else None,
                    description=metadata.get("description") if metadata else None,
                    duration=metadata.get("duration") if metadata else None,
                )
                loop = asyncio.new_event_loop()
                try:
                    lyric_result = loop.run_until_complete(fetch_lyrics(video_info, job_log=jl))
                finally:
                    loop.close()
                if lyric_result.get("lines"):
                    jl.info(
                        f"Auto-fetched {len(lyric_result['lines'])} lyrics lines "
                        f"via {lyric_result.get('provider', 'unknown')}"
                    )
            except Exception as lyr_exc:
                jl.warning(f"Auto lyrics fetch failed (non-fatal): {lyr_exc}")

        # ── Optional step 5: offline DTW match + write canonical output ──────────
        # Runs only if lyrics_cached.json exists (from the lyrics provider system).
        # Selection happens once here; canonical_lines.json is the permanent result.
        # GET /jobs/{id}/result reads canonical_lines.json — no re-selection on reads.
        try:
            match_result = run_offline_match(job_cache, yt_id, job_log=jl)
            if match_result:
                jl.info(
                    f"Offline match: {len(match_result['lines'])} lines, "
                    f"avg_conf={match_result['stats'].get('avgConfidence', 0):.2f}, "
                    f"avg_sim={match_result['stats'].get('avgSimilarity', 0):.2f}",
                )
            _write_canonical(job_cache, jl)
        except Exception as match_exc:
            jl.warning(f"Offline match/canonical skipped: {match_exc}")

        result_path = str(job_cache / "lyrics.json")
        set_job_status(job_id, JobStatus.completed, result_path=result_path)
        jl.info(
            f"Job complete → {len(_lines)} lines, "
            f"confidence={confidence.get('overall', 0):.2f}, "
            f"backend={backend}, method={align_method}"
        )

    except Exception as exc:
        jl.error(f"Job failed: {exc}")
        logger.exception("[%s] Job failed", job_id)
        set_job_status(job_id, JobStatus.failed, error=str(exc))


def _mark_all_complete(job_id: str) -> None:
    for step, (_, hi) in _WEIGHTS.items():
        update_job_step(job_id, step, JobStatus.completed, 100, hi)
