"""
Visual timing normalization + timeline finalization.

Two functions, called in sequence from steps/align.py:

  1. visual_timing_normalization(lines)
     Overwrites startTime/endTime with display-optimised values.
     Saves acoustic originals as acousticStart/acousticEnd (metadata only).

  2. finalize_timeline(lines)
     Enforces strict monotonicity and no overlaps.
     After this call, startTime/endTime are immutable.

Design principles:
  - Single source of truth: one startTime, one endTime per line.
  - displayEnd targets perceived vocal completion: word anchors are blended with
    segment boundaries and karaoke-natural linger to avoid premature line exits.
    A word anchor may trim at most _MAX_WORD_SAVE_S from acousticEnd.
  - Instrumental gaps are not absorbed into neighbouring lyric windows.
  - Mora-based duration compression prevents 15s display windows.
  - All logic runs offline. The playback layer reads times and does nothing else.
"""
from __future__ import annotations

import unicodedata

# ── Constants ─────────────────────────────────────────────────────────────────

_MORA_PER_SEC         = 0.22   # expected seconds per mora in sung Japanese
_VIS_MIN_S            = 1.2    # hard minimum display duration
_MORA_FLOOR_PER_MORA  = 0.15   # Phase E: mora-aware floor (seconds per mora)
_VIS_MAX_S            = 6.5    # maximum display duration
_INST_GAP_THRESH_S      = 4.0   # gap > this = instrumental break
_VOCAL_GAP_THRESH_S     = 1.5   # gap > this = notable vocal pause
_SAFE_OVERHANG_CONT_S   = 0.8   # perceptual hold: near-contiguous vocal phrase (gap 0.1–1.5s)
_SAFE_OVERHANG_SOFT_S   = 0.5   # sustain decay: phrase/verse transition (gap 1.5–4s)
_SAFE_OVERHANG_STRONG_S = 0.35  # brief tail: before instrumental break (gap >4s)
_SMALL_KANA = frozenset('ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ')

# ── Karaoke continuity constants ──────────────────────────────────────────────
_CONTIGUOUS_THRESH_S      = 0.10  # gap ≤ this → vocally contiguous (no break)
_NEAR_CONTIGUOUS_THRESH_S = 1.5   # gap ≤ this → vocal pause (breath/reset allowed)
_CONTIGUOUS_ONSET_RATIO   = 2.0   # contiguous lines: skip compression below this ratio
_SUSTAIN_MULT_CONTIGUOUS  = 0.20  # strength multiplier for contiguous pairs
_SUSTAIN_MULT_NEAR        = 0.50  # strength multiplier for near-contiguous pairs
_SUSTAIN_MULT_SOFT        = 0.70  # strength multiplier for phrase-boundary lines (SOFT gap ahead)
_VOCAL_OVERLAP_S          = 0.5   # anchor may extend this far past next_start (vocal tail)
_MIN_WORD_CONFIDENCE      = 0.6   # RULE 1: minimum WhisperX score to accept last_word anchor
_MIN_END_BUFFER_S         = 0.15  # RULE 2: hard minimum buffer past word anchor end
_ANTI_COMPRESS_RATIO      = 2.0   # RULE 6: skip compression below this eff_ratio (word-anchored)
_MAX_START_PUSH_S         = 0.5   # RULE 5: max overlap resolved by pushing next line's start
_SEGMENT_OVERRIDE_GAP_S   = 0.6   # Ref 1: segment_end overrides last_word if tail gap exceeds this
_SEGMENT_OVERRIDE_MAX_CONF= 0.85  # Ref 1: segment_end override only when confidence below this
_WORD_COMPRESS_MULT       = 0.6   # Ref 2: compression strength multiplier for word-anchored near-target lines
_DRIFT_GUARD_THRESH       = 0.05  # Ref 3: max allowed cumulative extension as fraction of acoustic total
# DTW-tail relaxation: widen segment_end ceiling for CONT lines where DTW allocated
# significantly more territory than the last WhisperX word covers.
_DTW_TAIL_RELAX_THRESH_S  = 0.8   # min dtw_tail to trigger ceiling relaxation
_DTW_TAIL_RELAX_SCALE     = 0.75  # fraction of dtw_tail added past next_start
_DTW_TAIL_RELAX_MAX_S     = 2.5   # max extension beyond next_start

# ── Phase 3: karaoke pacing ───────────────────────────────────────────────────
_MAX_WORD_SAVE_S         = 1.0   # max seconds a word anchor may trim from acousticEnd
_DTW_TAIL_BONUS_THRESH_S = 3.0   # dtw_tail above which segment_end reward cap unlocks
_SEG_END_REWARD_CAP      = 4.0   # normal tail reward cap for segment_end
_SEG_END_REWARD_CAP_L    = 5.0   # large-dtw-tail tail reward cap (beats high-conf last_word)
_LINGER_BASE_S           = 0.25  # base karaoke linger for non-contiguous lines
_LINGER_MAX_S            = 0.80  # maximum linger ceiling
_LINGER_PRESSURE_DIST    = 0.80  # gap below which linger is progressively suppressed
_VOWEL_END_CHARS         = frozenset('あいうえおアイウエオーaeiouAEIOU')
_FINAL_HOLD_PUNCT        = frozenset('？！…～!?')

# ── Adaptive display cap (replaces universal hard cap) ────────────────────────
# Per-line cap derived from mora count, following gap, and vocal characteristics.
# Typical lines: 4–7 s.  Emotional endings: 7–10 s.  Chorus climax: 10–14 s.
_MORA_SUNG_S            = 0.40   # sec/mora for sung Japanese (used in words=0 fallback)
_CAP_BASE_S             = 4.0    # base cap for any line
_CAP_MORA_SLOPE         = 0.30   # extra cap seconds per mora (longer text → more room)
_CAP_FINAL_HOLD_S       = 1.5    # bonus: phrase-final vowel or exclamatory punctuation
_CAP_STRONG_BREAK_S     = 1.0    # bonus: mid-song STRONG gap (instrumental break)
_CAP_SONG_FINAL_S       = 5.0    # bonus: last line or gap > 20 s (song-final climax)
_CAP_SOFT_BREAK_S       = 0.5    # bonus: SOFT vocal pause following this line
_CAP_MIN_S              = 4.0    # minimum adaptive cap
_CAP_MAX_S              = 14.0   # maximum adaptive cap
_NO_WORDS_SHORT_THRESH  = 8      # mora ≤ this → short wordless line gets a sung floor
_NO_WORDS_SHORT_FLOOR   = 0.68   # sec/mora floor for short wordless lines
_SUSTAIN_MULT_SOFT_HIGH = 0.90   # SOFT + line_ratio > 2.5 → stronger compression

# ── Onset correction ───────────────────────────────────────────────────────────
# Separates acoustic_start (VAD boundary) from vocal_onset (first voiced content)
# so anti-stall budget and display start are anchored to actual singing, not to
# the segment boundary that may include breath, sparse piano, or pre-vocal silence.
_ONSET_MIN_SHIFT_S      = 0.05  # ignore shifts smaller than this (noise floor)
_ONSET_MAX_SHIFT_S      = 3.0   # vad_anchor: max shift from acoustic_start
_FIRST_WORD_MAX_SHIFT_S = 6.0   # first_word: accepts larger DTW-split gaps (CTC timestamps reliable)
_ONSET_PRELOAD_S        = 0.10  # UX lead — line is visible just before first word


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mora_count(text: str) -> int:
    count = 0
    for ch in unicodedata.normalize('NFKC', text):
        cp = ord(ch)
        if ch in _SMALL_KANA:
            pass
        elif 0x3041 <= cp <= 0x3096 or 0x30A1 <= cp <= 0x30F6:
            count += 1
        elif ch in ('ー', '〜'):
            count += 1
        elif 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            count += 2
        elif ch.isascii() and ch.isalpha():
            count += 1
    return max(1, count)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def _has_vowel_tail(text: str) -> bool:
    """True when the last non-punctuation character is a Japanese vowel or elongation mark."""
    for ch in reversed(text):
        if not ch.isspace() and ch not in '。、！？!?.…':
            return ch in _VOWEL_END_CHARS
    return False


def _adaptive_hard_cap(
    mora: int,
    gap_to_next: float,
    is_last_line: bool,
    has_no_words: bool,
    has_final_hold: bool,
) -> float:
    """
    Compute a per-line adaptive display cap in seconds.

    Replaces the universal 10 s hard cap with a musically-aware limit:
      - Short text exits faster (fewer moras → smaller cap)
      - Phrase-final holds get extra room
      - Chorus-climax / song-final lines can reach 14 s
      - words=0 lines are tightened to their mora estimate (prevents runaway segments)
    """
    cap = _CAP_BASE_S + mora * _CAP_MORA_SLOPE

    if has_final_hold:
        cap += _CAP_FINAL_HOLD_S

    if is_last_line or gap_to_next > 20.0:
        cap += _CAP_SONG_FINAL_S
    elif gap_to_next > _INST_GAP_THRESH_S:        # STRONG mid-song break
        cap += _CAP_STRONG_BREAK_S
    elif gap_to_next > _VOCAL_GAP_THRESH_S:       # SOFT vocal pause
        cap += _CAP_SOFT_BREAK_S

    # When there are no word timestamps, the WhisperX segment boundary is
    # unreliable.  Tighten to a mora-based estimate (unless the line ends on
    # a sustained vowel, in which case we trust the acoustic end more).
    if has_no_words and not has_final_hold:
        mora_estimate = mora * _MORA_SUNG_S          # e.g. 6 mora → 2.4 s
        cap = min(cap, mora_estimate * 1.3)

    return max(_CAP_MIN_S, min(_CAP_MAX_S, cap))


def _compression_strength(ratio: float) -> float:
    """
    Acoustic-to-target ratio → compression strength [0, 1].

    More aggressive than the old lerp curve — designed to prevent 10–15s
    display windows while still preserving musical phrasing at moderate ratios.

    Breakpoints (ratio → strength):
      ≤ 1.2   →  0.00   (no compression, within range)
      1.2–1.8 →  0.00–0.35  (light, onset of compression)
      1.8–2.5 →  0.35–0.85  (moderate to strong, main working range)
      2.5–4.0 →  0.85–0.93  (strong, sustain-heavy lines)
      4.0–6.0 →  0.93–0.97  (very strong, near-maximum)
      > 6.0   →  0.97   (maximum)
    """
    if ratio <= 1.2:   return 0.0
    elif ratio <= 1.8: return _lerp(0.00, 0.35, (ratio - 1.2) / 0.6)
    elif ratio <= 2.5: return _lerp(0.35, 0.85, (ratio - 1.8) / 0.7)
    elif ratio <= 4.0: return _lerp(0.85, 0.93, (ratio - 2.5) / 1.5)
    elif ratio <= 6.0: return _lerp(0.93, 0.97, (ratio - 4.0) / 2.0)
    else:              return 0.97


# ── Phrase / boundary model ───────────────────────────────────────────────────

def classify_gap(gap_s: float) -> str:
    """
    Classify an inter-line gap by its musical role.

      STRONG  gap > _INST_GAP_THRESH_S  – instrumental break or bridge section
      SOFT    gap > _VOCAL_GAP_THRESH_S – phrase or verse transition (short pause)
      CONT    otherwise                  – within-phrase connection; no structural break
    """
    if gap_s > _INST_GAP_THRESH_S:
        return 'STRONG'
    elif gap_s > _VOCAL_GAP_THRESH_S:
        return 'SOFT'
    return 'CONT'


def _segment_phrases(
    lines: list[dict],
) -> tuple[list[str], dict[int, float]]:
    """
    Pre-pass: classify every inter-line gap and compute a phrase-level acoustic
    budget ratio for each line.

    Phrases are contiguous runs of lines separated only by CONT boundaries.
    SOFT or STRONG gaps end a phrase.  The phrase ratio (total acoustic duration
    divided by total mora-based target) caps per-line compression: a short-text
    line that sits in a phrase with long-text anchor lines gets softer compression
    rather than being crushed to its individual target.

    Returns:
      gap_types    list[str] of length n-1, boundary type for gap[i→i+1]
      phrase_ratios dict mapping each line index to its phrase's acoustic ratio
    """
    n = len(lines)
    if n == 0:
        return [], {}

    # Classify all adjacent gaps using acoustic (pre-normalisation) timing
    gap_types: list[str] = []
    for i in range(n - 1):
        a_end     = float(lines[i].get('endTime', 0))
        nxt_start = float(lines[i + 1].get('startTime', a_end))
        gap_types.append(classify_gap(max(0.0, nxt_start - a_end)))

    # Group lines into phrases (phrase ends at each SOFT/STRONG boundary)
    phrases: list[list[int]] = []
    current: list[int] = [0]
    for i, btype in enumerate(gap_types):
        if btype in ('SOFT', 'STRONG'):
            phrases.append(current)
            current = [i + 1]
        else:
            current.append(i + 1)
    phrases.append(current)

    # Compute acoustic/target ratio for each phrase, assign to every member line
    phrase_ratios: dict[int, float] = {}
    for phrase_idxs in phrases:
        total_acoustic = sum(
            max(0.0, float(lines[i].get('endTime', 0)) - float(lines[i].get('startTime', 0)))
            for i in phrase_idxs
        )
        total_target = sum(
            max(_VIS_MIN_S, min(_VIS_MAX_S, _mora_count(lines[i].get('text', '')) * _MORA_PER_SEC))
            for i in phrase_idxs
        )
        p_ratio = total_acoustic / total_target if total_target > 0 else 1.0
        for i in phrase_idxs:
            phrase_ratios[i] = p_ratio

    return gap_types, phrase_ratios


def pick_anchor(
    words:           list,
    a_start:         float,
    a_end:           float,
    overlap_ceiling: float,
    job_log=None,
    line_idx:        int = -1,
    seg_ceiling:     float | None = None,
) -> tuple[float, str]:
    """
    Unified candidate pool anchor selection.

    All candidates (last_word, second_last_word, segment_end) are always
    built and scored independently.  The highest-scoring valid candidate wins.

    Word candidates use overlap_ceiling as their upper bound.
    segment_end uses seg_ceiling when provided (DTW-tail relaxation), otherwise
    falls back to overlap_ceiling.  This lets segment_end reach DTW-allocated
    territory without loosening the ceiling for word candidates.

    Validity (word candidates): a_start + VIS_MIN_S*0.5 < t <= min(overlap_ceiling, a_end+0.1)
    Validity (segment_end):     t > a_start + VIS_MIN_S*0.5  (upper bound is seg_ceiling itself)

    Additive scoring
    ─────────────────────────────────────────────────────────────────
    last_word        base +5.0
                     conf_bonus  +conf × 1.5
                     moderate-conf (0.6 ≤ c < 0.85)  −2.0
                     low-conf     (c < 0.6)           −4.0
                     short-token  (dur < 50 ms)       −1.5

    second_last_word base +3.0
                     conf_bonus  +conf × 0.5

    segment_end      base +2.0
                     tail_reward (tail > 0.6 s)  +min(tail × 3.5, 4.0)
                     no-words boost              +2.0
    """
    hard_floor   = a_start + _VIS_MIN_S * 0.5
    hard_ceiling = min(overlap_ceiling, a_end + 0.1)   # word candidates only

    def _valid(t: float) -> bool:
        return hard_floor < t <= hard_ceiling

    candidates: list[tuple[float, str, float]] = []

    # ── last_word ─────────────────────────────────────────────────────────────
    if words:
        last     = words[-1]
        last_end = last.get('end')
        if last_end is not None:
            last_end = float(last_end)
            if _valid(last_end):
                conf = float(last.get('score', 1.0))
                dur  = last_end - float(last.get('start', last_end))
                sc   = 5.0 + conf * 1.5
                if conf < _MIN_WORD_CONFIDENCE:
                    sc -= 4.0
                elif conf < _SEGMENT_OVERRIDE_MAX_CONF:
                    sc -= 2.0
                if dur < 0.05:
                    sc -= 1.5
                candidates.append((last_end, 'last_word', sc))
                if job_log:
                    job_log.info(
                        f"[anchor_pool] L{line_idx:02d} last_word: "
                        f"t={last_end:.3f}s conf={conf:.2f} dur={dur:.3f}s score={sc:.2f}",
                        stage="align",
                    )

    # ── second_last_word ──────────────────────────────────────────────────────
    if len(words) >= 2:
        sl     = words[-2]
        sl_end = sl.get('end')
        if sl_end is not None:
            sl_t = min(float(sl_end) + 0.2, overlap_ceiling)
            if _valid(sl_t):
                conf = float(sl.get('score', 1.0))
                sc   = 3.0 + conf * 0.5
                candidates.append((sl_t, 'second_last_word', sc))
                if job_log:
                    job_log.info(
                        f"[anchor_pool] L{line_idx:02d} second_last_word: "
                        f"t={sl_t:.3f}s conf={conf:.2f} score={sc:.2f}",
                        stage="align",
                    )

    # ── segment_end ───────────────────────────────────────────────────────────
    # Uses seg_ceiling (DTW-aware relaxed ceiling) when available; otherwise the
    # same overlap_ceiling as word candidates.  Upper bound is implicit in how
    # seg_end is computed: min(a_end, _seg_ceil) is always ≤ _seg_ceil.
    _seg_ceil = seg_ceiling if seg_ceiling is not None else overlap_ceiling
    seg_end   = min(a_end, _seg_ceil)
    if seg_end > hard_floor:
        sc = 2.0
        if words:
            lw_end = words[-1].get('end')
            if lw_end is not None:
                tail = seg_end - float(lw_end)
                if tail > _SEGMENT_OVERRIDE_GAP_S:
                    # When the DTW boundary extends far past the last word, allow
                    # segment_end to outscore even high-confidence last_word anchors.
                    _dt = (a_end - float(lw_end)) if seg_ceiling is not None else 0.0
                    _cap = _SEG_END_REWARD_CAP_L if _dt > _DTW_TAIL_BONUS_THRESH_S else _SEG_END_REWARD_CAP
                    sc += min(tail * 3.5, _cap)
        else:
            sc += 2.0  # no word timestamps at all → boost segment_end
        candidates.append((seg_end, 'segment_end', sc))
        if job_log:
            job_log.info(
                f"[anchor_pool] L{line_idx:02d} segment_end: "
                f"t={seg_end:.3f}s score={sc:.2f}"
                + (f" [seg_ceil={_seg_ceil:.3f}s]" if seg_ceiling is not None else ""),
                stage="align",
            )

    # ── Select best ───────────────────────────────────────────────────────────
    if not candidates:
        fallback = min(a_end, overlap_ceiling)
        if job_log:
            job_log.info(
                f"[anchor_pool] L{line_idx:02d} → fallback={fallback:.3f}s (no valid candidates)",
                stage="align",
            )
        return fallback, 'segment_end'

    chosen_t, chosen_src, _ = max(candidates, key=lambda c: c[2])
    return chosen_t, chosen_src


# ── Phase 1: visual normalization ─────────────────────────────────────────────

def visual_timing_normalization(lines: list[dict], job_log=None) -> list[dict]:
    """
    Overwrite startTime/endTime with display-optimised values.

    Acoustic originals preserved as acousticStart / acousticEnd.
    After this call, startTime/endTime reflect display timing only.
    finalize_timeline() must be called after this to enforce monotonicity.

    Karaoke continuity guarantee:
      Lines that were acoustically contiguous (gap ≤ _CONTIGUOUS_THRESH_S) remain
      visually contiguous after normalization.  Compression on those lines is heavily
      reduced (strength × _SUSTAIN_MULT_CONTIGUOUS) because their "excess" duration
      is sustained vocal, not silence.  After computing display_end, the next line's
      startTime is pulled forward to match — preventing the normalizer from
      manufacturing dead air inside a vocal phrase.
    """
    n = len(lines)

    # ── Phrase segmentation pre-pass ───────────────────────────────────────────
    # Classify every inter-line gap and compute per-phrase acoustic budget ratios.
    # gap_types[i]   = boundary type for the gap between line i and line i+1
    # phrase_ratios[i] = phrase acoustic / phrase mora-target ratio for line i
    _gap_types, _phrase_ratios = _segment_phrases(lines)

    n_word_anchored  = 0
    n_compressed     = 0
    n_sustain        = 0
    n_inst_capped    = 0
    n_continuity     = 0
    n_pacing_clamped = 0
    total_saved      = 0.0

    for i, line in enumerate(lines):
        a_start = float(line.get('startTime', 0))
        a_end   = float(line.get('endTime',   0))
        a_dur   = max(0.0, a_end - a_start)

        # Boundary type for the gap TO the next line (pre-computed from acoustic timing)
        boundary_type = _gap_types[i] if i < len(_gap_types) else 'END'

        # Preserve acoustic originals
        line['acousticStart'] = round(a_start, 3)
        line['acousticEnd']   = round(a_end,   3)

        # ── Vocal continuity detection ─────────────────────────────────────
        # Read next.startTime BEFORE any propagation mutation.
        next_acoustic_start  = float(lines[i + 1].get('startTime', 9999.0)) if i + 1 < n else 9999.0
        gap_to_next_acoustic = next_acoustic_start - a_end
        is_contiguous        = gap_to_next_acoustic <= _CONTIGUOUS_THRESH_S
        is_near_contiguous   = (not is_contiguous) and gap_to_next_acoustic <= _NEAR_CONTIGUOUS_THRESH_S

        # ── Timing anchor: unified candidate pool ─────────────────────────
        overlap_ceiling = next_acoustic_start + _VOCAL_OVERLAP_S
        _line_words     = line.get('words') or []

        # ── Phase 0: vocal onset estimation ───────────────────────────────────
        # Determines how far startTime should be pushed forward from acoustic_start
        # to align with the first voiced content rather than the VAD boundary.
        #
        # Source priority:
        #   1. _vad_anchor_ms (saved by aligner) — offset between Whisper VAD
        #      segment start and the first CTC word timestamp.  This is the most
        #      reliable signal: it directly measures pre-vocal silence + CTC lag
        #      for segments where _apply_vad_anchor was applied.
        #   2. first_word.start — only non-trivial when VAD anchor was *skipped*
        #      (offset ≥ _MAX_CTC_LAG_S) and word timestamps retain their original
        #      CTC positions, which happen to be close to the actual vocal onset.
        #   3. acoustic_start — fallback when neither signal provides useful info.
        _vad_anchor_ms = float(line.get('_vad_anchor_ms', 0.0))
        _onset_source  = 'acoustic'
        _vocal_onset   = a_start
        _fw_start_log: float | None = None

        if _vad_anchor_ms >= _ONSET_MIN_SHIFT_S * 1000:
            _candidate = a_start + _vad_anchor_ms / 1000.0
            if _ONSET_MIN_SHIFT_S <= (_candidate - a_start) <= _ONSET_MAX_SHIFT_S:
                _vocal_onset  = _candidate
                _onset_source = 'vad_anchor'

        if _onset_source == 'acoustic' and _line_words:
            for _w in _line_words:
                _ws = _w.get('start')
                if _ws is not None:
                    _fw_start_log = float(_ws)
                    _shift = _fw_start_log - a_start
                    if _ONSET_MIN_SHIFT_S <= _shift <= _FIRST_WORD_MAX_SHIFT_S:
                        _vocal_onset  = _fw_start_log
                        _onset_source = 'first_word'
                    break

        # Phase 0 third fallback: early-word stretch scanner.
        # Scans words k=0..3 for a word whose duration is anomalously long relative
        # to the following words — indicating CTC absorbed pre-vocal silence or a
        # cross-boundary bleed tail into that word's span.  Generalises the original
        # collapsed-first-word heuristic (k=0 only) to positions k=1, 2, 3, catching
        # bleed-tail patterns where the stretched word is not the first in the line.
        if _onset_source == 'acoustic' and len(_line_words) >= 3:
            for _ews_k in range(min(4, len(_line_words) - 2)):
                _ews_w    = _line_words[_ews_k]
                _ews_wend = _ews_w.get('end')
                if _ews_w.get('start') is None or _ews_wend is None:
                    continue
                # Bleed-tail patterns only occur when very little real lyric content
                # exists before the stretched word. If substantial sung content already
                # occurred, this is likely a legitimate held note inside the phrase.
                _ews_prior_dur = sum(
                    float(_line_words[j].get('end', 0)) - float(_line_words[j].get('start', 0))
                    for j in range(_ews_k)
                    if _line_words[j].get('start') is not None
                    and _line_words[j].get('end') is not None
                )
                if _ews_prior_dur > 1.5:
                    break
                _ews_dur = float(_ews_wend) - float(_ews_w['start'])
                if _ews_dur <= 1.2:
                    continue
                _ews_next_durs = [
                    float(w['end']) - float(w['start'])
                    for w in _line_words[_ews_k + 1 : _ews_k + 5]
                    if w.get('start') is not None and w.get('end') is not None
                ]
                if not _ews_next_durs:
                    continue
                _ews_sorted      = sorted(_ews_next_durs)
                _ews_median_next = _ews_sorted[len(_ews_sorted) // 2]
                _ews_ratio       = _ews_dur / _ews_median_next if _ews_median_next > 0 else 0.0
                if _ews_median_next >= 0.12 and _ews_ratio > 3.0:
                    _ews_end_f = float(_ews_wend)
                    _ews_onset = _ews_end_f - min(_ews_median_next * 1.2, _ews_dur * 0.45)
                    _ews_onset = min(_ews_onset, _ews_end_f - 0.12)
                    _ews_shift = _ews_onset - a_start
                    if _ONSET_MIN_SHIFT_S <= _ews_shift <= 6.0:
                        _vocal_onset  = _ews_onset
                        _onset_source = 'early_word_stretch'
                        if job_log:
                            job_log.info(
                                f"[early_word_stretch] L{i:02d}: "
                                f"k={_ews_k} "
                                f"word={_ews_w.get('word', '?')!r} "
                                f"duration={_ews_dur:.3f}s "
                                f"median_next={_ews_median_next:.3f}s "
                                f"ratio={_ews_ratio:.2f} "
                                f"inferred_onset={_vocal_onset:.3f}s "
                                f"shift_ms={round(_ews_shift * 1000):.0f}",
                                stage="align",
                            )
                        break

        # Preload: line activates _ONSET_PRELOAD_S before vocal onset so the UI
        # has time to scroll and the viewer sees the line just before singing.
        # Never before acoustic_start (no audio signal exists before that point).
        _display_start  = max(a_start, _vocal_onset - _ONSET_PRELOAD_S)
        _onset_shift_ms = round((_vocal_onset - a_start) * 1000.0, 1)

        # DTW-tail relaxation: when a CONT line's acousticEnd extends significantly
        # past the last WhisperX word, allow segment_end to reach into the DTW tail.
        # Word candidates (last_word, second_last_word) always use overlap_ceiling.
        seg_ceiling = overlap_ceiling  # default: same tight ceiling for all
        if boundary_type == 'CONT' and _line_words:
            _lw = _line_words[-1]
            _lw_end = _lw.get('end')
            if _lw_end is not None:
                dtw_tail = a_end - float(_lw_end)
                if dtw_tail > _DTW_TAIL_RELAX_THRESH_S:
                    seg_ceiling = min(
                        a_end,
                        next_acoustic_start + min(dtw_tail * _DTW_TAIL_RELAX_SCALE, _DTW_TAIL_RELAX_MAX_S),
                    )
                    if job_log:
                        job_log.info(
                            f"[dtw_tail_relax] L{i:02d}: "
                            f"last_word={float(_lw_end):.3f}s acousticEnd={a_end:.3f}s "
                            f"dtw_tail={dtw_tail:.3f}s "
                            f"orig_ceiling={overlap_ceiling:.3f}s "
                            f"relaxed_ceiling={seg_ceiling:.3f}s",
                            stage="align",
                        )

        anchor_end, anchor_src = pick_anchor(
            _line_words, a_start, a_end, overlap_ceiling,
            job_log=job_log, line_idx=i,
            seg_ceiling=seg_ceiling if seg_ceiling != overlap_ceiling else None,
        )
        if 'word' in anchor_src:
            n_word_anchored += 1
        if job_log:
            job_log.info(
                f"[anchor] L{i:02d}: winner={anchor_src} → t={anchor_end:.3f}s "
                f"(acousticEnd={a_end:.3f}s ceiling={overlap_ceiling:.3f}s "
                + (f"seg_ceiling={seg_ceiling:.3f}s " if seg_ceiling != overlap_ceiling else "")
                + f"words={len(_line_words)})",
                stage="align",
            )

        # ── Early text signals (needed before word-save clamp) ────────────
        text           = line.get('text', '')
        has_no_words   = len(_line_words) == 0
        moras          = _mora_count(text)
        has_vowel_tail = _has_vowel_tail(text)
        _text_strip    = text.rstrip()
        has_final_hold = has_vowel_tail or bool(_text_strip and _text_strip[-1] in _FINAL_HOLD_PUNCT)

        # ── words=0: mora-based anchor override ───────────────────────────
        # When WhisperX produced no word timestamps the segment_end boundary
        # is unreliable (often a fused multi-line segment).  Replace it with
        # a mora-based estimate of vocal duration.  Lines ending on a sustained
        # vowel keep the acoustic end (singer is still holding; trust WhisperX).
        if has_no_words and anchor_src == 'segment_end' and not has_final_hold:
            mora_anchor_end = a_start + moras * _MORA_SUNG_S
            anchor_end      = min(anchor_end, mora_anchor_end)
            # Short wordless lines: also enforce a sung minimum so they don't
            # under-display (e.g. 「聞こえるかな」 6 mora → floor ≈ 4 s).
            if moras <= _NO_WORDS_SHORT_THRESH:
                anchor_end = max(anchor_end, a_start + moras * _NO_WORDS_SHORT_FLOOR)
            anchor_src = 'mora_estimate'

        # ── Phase 3: word-save clamp ──────────────────────────────────────────
        # No word anchor may trim more than _MAX_WORD_SAVE_S from acousticEnd.
        # This preserves sustained vowels / melodic tails that WhisperX timestamps
        # early. effective_anchor_end replaces anchor_end in all timing math below.
        word_save_raw     = max(0.0, a_end - anchor_end) if 'word' in anchor_src else 0.0
        word_save_clamped = word_save_raw > _MAX_WORD_SAVE_S
        effective_anchor_end = (a_end - _MAX_WORD_SAVE_S) if word_save_clamped else anchor_end
        if word_save_clamped:
            n_pacing_clamped += 1

        # Phase 3: sustained-singing indicators (for bias detection and logging)
        has_dtw_relax     = seg_ceiling != overlap_ceiling
        has_sparse_words  = 0 < len(_line_words) <= 3
        sustain_bias_used = word_save_clamped and (has_vowel_tail or has_dtw_relax or has_sparse_words)

        anchor_dur = max(0.0, effective_anchor_end - a_start)

        # ── Mora-based visual target + mora-aware minimum ──────────────────
        vis_target   = max(_VIS_MIN_S, min(_VIS_MAX_S, moras * _MORA_PER_SEC))
        mora_min_dur = max(_VIS_MIN_S, moras * _MORA_FLOOR_PER_MORA)
        ratio        = anchor_dur / vis_target if vis_target > 0 else 1.0

        # ── Phrase budget ceiling ──────────────────────────────────────────
        # For CONT lines: cap compression at phrase-level ratio so short-text
        # lines in a long-text phrase are not over-compressed.
        # For SOFT/STRONG/END lines: these ARE the phrase-final lines; apply
        # their full line ratio so phrasing exits compress as needed.
        phrase_ratio    = _phrase_ratios.get(i, ratio)
        if boundary_type in ('SOFT', 'STRONG', 'END'):
            effective_ratio = ratio
        else:
            effective_ratio = min(ratio, phrase_ratio)

        transition_bias = 0.0

        # ── Compression with sustain-awareness ────────────────────────────
        # Uses effective_ratio (phrase-budget-capped) rather than line ratio,
        # so compression stays proportional to the phrase's overall over-budget.
        has_words   = bool(_line_words)
        onset_ratio = _CONTIGUOUS_ONSET_RATIO if is_contiguous else 1.1

        # Refinement 2: reduce (not disable) compression for word-anchored lines that
        # are near their mora target. Applied as a multiplier on eff_strength.
        word_compress_mult = (_WORD_COMPRESS_MULT
                              if has_words and effective_ratio < _ANTI_COMPRESS_RATIO
                              else 1.0)

        if effective_ratio <= onset_ratio:
            display_end   = max(a_start + mora_min_dur, effective_anchor_end - transition_bias)
            sustain_heavy = False
            raw_strength  = 0.0
            eff_strength  = 0.0
            strategy      = ('sustain_preserve' if is_contiguous
                             else 'phrase_end_strong' if boundary_type == 'STRONG'
                             else 'light')
        else:
            raw_strength = _compression_strength(effective_ratio)
            if is_contiguous:
                eff_strength = raw_strength * _SUSTAIN_MULT_CONTIGUOUS
                strategy     = 'sustain_preserve'
            elif is_near_contiguous:
                eff_strength = raw_strength * _SUSTAIN_MULT_NEAR
                strategy     = 'near_sustain'
            elif boundary_type == 'STRONG':
                eff_strength = raw_strength * _SUSTAIN_MULT_NEAR    # 0.50 — vocal tail before instrumental
                strategy     = 'phrase_end_strong'
            elif boundary_type == 'SOFT':
                if effective_ratio > 2.5:
                    # High-ratio SOFT: short text relative to long acoustic window.
                    # Use stronger compression so short lines (e.g. 「してたみたいに…」)
                    # exit quickly rather than lingering at 2× their mora duration.
                    eff_strength = raw_strength * _SUSTAIN_MULT_SOFT_HIGH  # 0.90
                    strategy     = 'phrase_end_soft_hi'
                else:
                    eff_strength = raw_strength * _SUSTAIN_MULT_SOFT       # 0.70
                    strategy     = 'phrase_end_soft'
            else:
                eff_strength = raw_strength
                strategy     = 'adaptive_curve'
            eff_strength  *= word_compress_mult   # Ref 2: 40% reduction when word-anchored + near target
            sustain_heavy  = effective_ratio > 4.0
            compressed_dur = _lerp(anchor_dur, vis_target, eff_strength)
            display_end    = a_start + compressed_dur - transition_bias
            display_end    = min(display_end, effective_anchor_end)  # base cap; perceptual overhang added below

        # ── Perceptual overhang: extend display_end past anchor for vocal tail ──
        # Word-anchor endpoints exit too early vs. perceived vocal completion.
        # Extend by a context-dependent buffer; next line's acoustic start is the
        # hard ceiling so we never visually overlap another line's territory.
        inst_capped = False
        if is_contiguous:
            safe_overhang = 0.0              # continuity propagation handles the seam
        elif is_near_contiguous:
            safe_overhang = _SAFE_OVERHANG_CONT_S
        elif boundary_type == 'SOFT':
            safe_overhang = _SAFE_OVERHANG_SOFT_S
        elif boundary_type == 'STRONG':
            safe_overhang = _SAFE_OVERHANG_STRONG_S
            inst_capped = True
            n_inst_capped += 1
        else:  # END (last line)
            safe_overhang = _SAFE_OVERHANG_SOFT_S

        phrase_boundary_cap = next_acoustic_start if i + 1 < n else effective_anchor_end + safe_overhang
        perceptual_end = min(effective_anchor_end + safe_overhang, phrase_boundary_cap)
        display_end = max(display_end, perceptual_end)

        # ── RULE 2: hard word-anchor floor ────────────────────────────────
        # display_end must never fall below anchor_end + _MIN_END_BUFFER_S
        # when the anchor is a real word timestamp (not segment_end).
        if anchor_src in ('last_word', 'second_last_word'):
            lw_floor = (float(_line_words[-1]['end']) + 0.15
                        if len(_line_words) >= 1 and 'end' in _line_words[-1] else anchor_end + 0.15)
            slw_floor = (float(_line_words[-2]['end']) + 0.1
                         if len(_line_words) >= 2 and 'end' in _line_words[-2] else 0.0)
            word_floor = min(max(lw_floor, slw_floor), phrase_boundary_cap)
            if display_end < word_floor:
                display_end = word_floor

        # ── Mora-aware minimum duration floor ─────────────────────────────
        if display_end < a_start + mora_min_dur:
            if job_log and mora_min_dur > _VIS_MIN_S:
                job_log.info(
                    f"[readability_floor] L{i:02d}: "
                    f"mora={moras} computed_min={mora_min_dur:.2f}s "
                    f"original={display_end - a_start:.2f}s "
                    f"adjusted={mora_min_dur:.2f}s",
                    stage="align",
                )
            display_end = a_start + mora_min_dur

        # ── Phase 3: hard save floor ──────────────────────────────────────────
        # After all floors and compression, guarantee display_end never loses more
        # than _MAX_WORD_SAVE_S from acousticEnd on word-anchored lines.
        if word_save_clamped and display_end < effective_anchor_end:
            display_end = effective_anchor_end

        # ── Phase 3: karaoke linger ───────────────────────────────────────────
        # Small adaptive tail added to all non-contiguous lines for musical feel.
        # Suppressed when the next phrase starts immediately.
        linger_added = 0.0
        if not is_contiguous:
            linger_s = _LINGER_BASE_S
            if has_vowel_tail:
                linger_s += 0.2
            if boundary_type in ('SOFT', 'END'):
                linger_s += 0.2
            elif boundary_type == 'STRONG':
                linger_s += 0.1
            linger_s = min(linger_s, _LINGER_MAX_S)
            headroom = phrase_boundary_cap - display_end
            if headroom < _LINGER_PRESSURE_DIST:
                linger_s *= max(0.0, headroom / _LINGER_PRESSURE_DIST)
            linger_applied = max(0.0, min(linger_s, headroom))
            if linger_applied > 0.01:
                display_end  += linger_applied
                linger_added  = round(linger_applied, 3)

        # ── Adaptive display cap ──────────────────────────────────────────────
        # Replaces the universal 10 s hard cap.  Each line gets a cap derived
        # from its mora count, following gap, and vocal characteristics.
        # Typical: 4–7 s.  Emotional endings: 7–10 s.  Chorus climax: 10–14 s.
        adaptive_cap     = _adaptive_hard_cap(
            mora=moras, gap_to_next=gap_to_next_acoustic,
            is_last_line=(i == n - 1), has_no_words=has_no_words,
            has_final_hold=has_final_hold,
        )
        adaptive_cap_end = a_start + adaptive_cap
        antistall_cap    = display_end > adaptive_cap_end
        if antistall_cap:
            if job_log:
                job_log.info(
                    f"[adaptive_cap] L{i:02d}: "
                    f"display={display_end - a_start:.2f}s → cap={adaptive_cap:.2f}s "
                    f"(mora={moras} gap={gap_to_next_acoustic:.1f}s "
                    f"final_hold={has_final_hold} no_words={has_no_words})",
                    stage="align",
                )
            display_end = adaptive_cap_end

        # Short wordless lines: enforce a sung-tempo minimum so they don't
        # under-display after mora-based anchor + compression shrinks them.
        # (e.g. 「聞こえるかな」 6 mora → floor = 4.1 s, manual ≈ 4 s)
        if has_no_words and moras <= _NO_WORDS_SHORT_THRESH:
            short_floor_end = a_start + moras * _NO_WORDS_SHORT_FLOOR
            if display_end < short_floor_end:
                display_end = short_floor_end

        saved = a_dur - (display_end - a_start)
        if saved > 0.1:
            n_compressed += 1
            total_saved  += saved
        if sustain_heavy:
            n_sustain += 1

        # Safety: display_start must leave at least _VIS_MIN_S of visible content.
        # If the onset shift is so large that it exceeds the display window (rare,
        # only for very short lines with a large pre-vocal silence), cap it.
        _display_start = min(_display_start, max(a_start, display_end - _VIS_MIN_S))

        # Onset instrumentation: log every line where a correction was applied.
        if job_log and _onset_shift_ms > 0:
            _fw_log = f"{_fw_start_log:.3f}s" if _fw_start_log is not None else "none"
            job_log.info(
                f"[onset_fix] L{i:02d}: "
                f"acoustic={a_start:.3f}s "
                f"first_word={_fw_log} "
                f"vocal_onset={_vocal_onset:.3f}s "
                f"display_start={_display_start:.3f}s "
                f"shift_ms={_onset_shift_ms:.0f} "
                f"has_words={bool(_line_words)} "
                f"source={_onset_source}",
                stage="align",
            )

        # Overwrite canonical times with display-optimised values.
        # startTime now reflects vocalOnset (not acoustic_start) so the frontend
        # anti-stall budget begins when singing starts, not at the VAD boundary.
        # acousticStart preserves the raw segment boundary for gap-detection logic.
        line['vocalOnset'] = round(_vocal_onset, 3)
        line['startTime']  = round(_display_start, 3)
        line['endTime']    = round(display_end, 3)

        # ── Continuity propagation (within-phrase only) ────────────────────
        # STRONG/SOFT boundaries never satisfy is_contiguous, so cross-phrase
        # propagation is structurally impossible — made explicit here.
        if is_contiguous and boundary_type == 'CONT' and i + 1 < n and display_end < next_acoustic_start:
            lines[i + 1]['startTime'] = round(display_end, 3)
            n_continuity += 1
            if job_log:
                job_log.info(
                    f"[continuity_propagated] L{i:02d}→L{i+1:02d}: "
                    f"next.start {next_acoustic_start:.3f} → {display_end:.3f}s "
                    f"(phrase-internal CONT gap={gap_to_next_acoustic:.3f}s)",
                    stage="align",
                )

        if job_log:
            flags = []
            if sustain_heavy:
                flags.append('sustain_heavy')
            if inst_capped:
                flags.append(f'inst-cap')
            if 'word' in anchor_src and word_save_raw > 0.5:
                if word_save_clamped:
                    flags.append(f'word-saved={word_save_raw:.1f}s→clamped')
                else:
                    flags.append(f'word-saved={word_save_raw:.1f}s')
            if phrase_ratio < ratio - 0.05:
                flags.append(f'phrase-cap(p={phrase_ratio:.2f}<l={ratio:.2f})')
            job_log.info(
                f"[visual_norm] L{i:02d} [{boundary_type}]: "
                f"acoustic={a_dur:.1f}s anchor={anchor_src}({anchor_dur:.1f}s) "
                f"target={vis_target:.1f}s eff_ratio={effective_ratio:.2f} "
                f"→ {a_start:.3f}–{display_end:.3f}s saved={saved:.1f}s"
                + (f" [{', '.join(flags)}]" if flags else ""),
                stage="align",
            )
            if effective_ratio > onset_ratio:
                job_log.info(
                    f"[compression] L{i:02d}: "
                    f"acoustic={anchor_dur:.2f}s target={vis_target:.2f}s "
                    f"line_ratio={ratio:.2f} phrase_ratio={phrase_ratio:.2f} eff_ratio={effective_ratio:.2f} "
                    f"strategy={strategy} raw={raw_strength:.3f} eff={eff_strength:.3f} "
                    f"final={display_end - a_start:.2f}s",
                    stage="align",
                )
            _cps = moras / max(display_end - a_start, 0.1)
            p3_parts = [
                f'cps={_cps:.1f}',
                f'mora={moras}',
                f'phrase_final_hold={has_final_hold}',
                f'no_words={has_no_words}',
                f'adaptive_cap={adaptive_cap:.1f}s',
                f'strategy={strategy}',
                f'final_dur={display_end - a_start:.2f}s',
            ]
            if word_save_clamped:
                p3_parts.append(f'word_save_clamped={word_save_raw:.2f}s')
            if sustain_bias_used:
                p3_parts.append('sustain_bias')
            if linger_added > 0.01:
                p3_parts.append(f'linger={linger_added:.3f}s')
            if antistall_cap:
                p3_parts.append('cap_applied')
            job_log.info(
                f"[pacing_v2] L{i:02d}: {' | '.join(p3_parts)}",
                stage="align",
            )

    # ── Refinement 3: timeline drift guard ────────────────────────────────────
    # If cumulative display extension exceeds _DRIFT_GUARD_THRESH (5%) of total
    # acoustic duration, scale each line's extension proportionally back to the
    # threshold. Start times are never moved; only end times are adjusted.
    _total_acoustic = sum(
        max(0.0, float(l.get('acousticEnd', l['endTime'])) - float(l.get('acousticStart', l['startTime'])))
        for l in lines
    )
    _total_display = sum(max(0.0, float(l['endTime']) - float(l['startTime'])) for l in lines)
    _excess = _total_display - _total_acoustic

    if _total_acoustic > 0 and _excess > _total_acoustic * _DRIFT_GUARD_THRESH:
        _target_excess   = _total_acoustic * _DRIFT_GUARD_THRESH
        _reduction_scale = _target_excess / _excess
        _n_drift = 0
        for _line in lines:
            _a_dur = max(0.0, float(_line.get('acousticEnd', _line['endTime']))
                              - float(_line.get('acousticStart', _line['startTime'])))
            _d_dur = max(0.0, float(_line['endTime']) - float(_line['startTime']))
            _ext   = max(0.0, _d_dur - _a_dur)
            if _ext > 0:
                _new_end = float(_line['startTime']) + _a_dur + _ext * _reduction_scale
                _line['endTime'] = round(_new_end, 3)
                _n_drift += 1
        if job_log:
            job_log.info(
                f"[drift_guard] excess={_excess:.2f}s "
                f"({100 * _excess / _total_acoustic:.1f}% of acoustic) "
                f"→ reduction_scale={_reduction_scale:.3f} adjusted={_n_drift} lines",
                stage="align",
            )

    # ── Onset correction summary ──────────────────────────────────────────────────
    _onset_shifts = [
        round((float(l.get('vocalOnset', l.get('acousticStart', l['startTime'])))
               - float(l.get('acousticStart', l['startTime']))) * 1000)
        for l in lines
    ]
    _n_onset_corrected = sum(1 for s in _onset_shifts if s > 0)
    _avg_onset_shift   = sum(_onset_shifts) / len(_onset_shifts) if _onset_shifts else 0.0
    if job_log:
        job_log.info(
            f"[onset_fix] summary: corrected={_n_onset_corrected}/{n} "
            f"avg_shift={_avg_onset_shift:.0f}ms "
            f"gt500ms={sum(1 for s in _onset_shifts if s > 500)} "
            f"gt1s={sum(1 for s in _onset_shifts if s > 1000)} "
            f"gt2s={sum(1 for s in _onset_shifts if s > 2000)}",
            stage="align",
        )

    if job_log:
        n_phrases = 1 + sum(1 for b in _gap_types if b in ('SOFT', 'STRONG'))
        n_strong  = sum(1 for b in _gap_types if b == 'STRONG')
        n_soft    = sum(1 for b in _gap_types if b == 'SOFT')
        n_cont    = sum(1 for b in _gap_types if b == 'CONT')
        avg_s = total_saved / n_compressed if n_compressed else 0.0
        job_log.info(
            f"[visual_norm] summary: phrases={n_phrases} "
            f"boundaries: {n_cont}xCONT {n_soft}xSOFT {n_strong}xSTRONG | "
            f"word_anchored={n_word_anchored}/{n} "
            f"compressed={n_compressed} sustain={n_sustain} "
            f"inst_capped={n_inst_capped} continuity_propagated={n_continuity} "
            f"pacing_clamped={n_pacing_clamped} "
            f"total_saved={total_saved:.1f}s avg={avg_s:.1f}s/line",
            stage="align",
        )

    return lines


# ── Phase 2: timeline finalization ────────────────────────────────────────────

def finalize_timeline(lines: list[dict], job_log=None) -> list[dict]:
    """
    Enforce a monotonic, non-overlapping timeline.

    Rules (applied in order):
      1. Sort by startTime (should already be sorted; this is a guarantee).
      2. If consecutive lines overlap (endTime[i] > startTime[i+1]),
         cap endTime[i] to startTime[i+1] - 50ms.
      3. Ensure every line has endTime > startTime + VIS_MIN_S.
      4. Log any adjustments.

    After this call, the timeline is immutable.
    Overlap resolution (RULE 5): for small overlaps (≤ _MAX_START_PUSH_S) push the
    next line's startTime forward rather than cutting the current line's endTime,
    preserving word-anchored vocal endings. Larger overlaps cap the current endTime.
    """
    if not lines:
        return lines

    lines.sort(key=lambda l: l.get('startTime', 0))

    n_capped   = 0
    n_extended = 0
    n_pushed   = 0

    for i, line in enumerate(lines):
        start = float(line['startTime'])
        end   = float(line['endTime'])

        # Enforce mora-aware minimum duration
        moras    = _mora_count(line.get('text', ''))
        min_dur  = max(_VIS_MIN_S, moras * _MORA_FLOOR_PER_MORA)
        if end < start + min_dur:
            new_end = round(start + min_dur, 3)
            if job_log and abs(new_end - end) > 0.01:
                job_log.info(
                    f"[finalize] L{i:02d}: extend {end:.3f} → {new_end:.3f}s "
                    f"(below mora-aware min {min_dur:.2f}s for {moras} mora)",
                    stage="align",
                )
            line['endTime'] = new_end
            n_extended += 1
            end = new_end

        # Overlap resolution (RULE 5): prefer pushing next line's start over
        # cutting the current line's end, up to _MAX_START_PUSH_S tolerance.
        if i + 1 < len(lines):
            next_start = float(lines[i + 1].get('startTime', end))
            if end > next_start:
                overlap = end - next_start
                if overlap <= _MAX_START_PUSH_S:
                    lines[i + 1]['startTime'] = round(end, 3)
                    n_pushed += 1
                    if job_log and overlap > 0.01:
                        job_log.info(
                            f"[finalize] L{i:02d}→L{i+1:02d}: pushed start "
                            f"{next_start:.3f} → {end:.3f}s (overlap={overlap:.3f}s)",
                            stage="align",
                        )
                else:
                    capped = round(next_start, 3)
                    capped = max(capped, start + _VIS_MIN_S)
                    if job_log and abs(capped - end) > 0.01:
                        job_log.info(
                            f"[finalize] L{i:02d}: cap {end:.3f} → {capped:.3f}s "
                            f"(large overlap={overlap:.3f}s L{i+1:02d} start={next_start:.3f}s)",
                            stage="align",
                        )
                    line['endTime'] = capped
                    n_capped += 1

    # Final monotonicity check (diagnostic only — never silently corrupt)
    n_nonmono = 0
    for i in range(1, len(lines)):
        if lines[i]['startTime'] <= lines[i - 1]['startTime']:
            n_nonmono += 1

    if job_log:
        status = "PASS" if n_nonmono == 0 else f"FAIL ({n_nonmono} non-monotonic starts)"
        job_log.info(
            f"[finalize] timeline locked: {len(lines)} lines | "
            f"capped={n_capped} pushed={n_pushed} extended={n_extended} monotonic={status}",
            stage="align",
        )

    return lines


# ── Quality summary ───────────────────────────────────────────────────────────

def timeline_quality(lines: list[dict], job_log=None) -> dict:
    """Emit a quality summary after finalization."""
    if not lines:
        return {}

    durs  = [max(0.0, l['endTime'] - l['startTime']) for l in lines]
    n     = len(lines)
    avg_d = sum(durs) / n
    n_8s  = sum(1 for d in durs if d > 8.0)
    n_5s  = sum(1 for d in durs if 5.0 < d <= 8.0)
    n_sh  = sum(1 for d in durs if d < 1.2)

    score  = max(0, 100 - n_8s * 15 - n_5s * 3 - n_sh * 5)
    grade  = ("EXCELLENT" if score >= 90 else "GOOD" if score >= 75
              else "FAIR" if score >= 55 else "POOR")

    metrics = {
        "lines": n, "avg_display_s": round(avg_d, 2),
        "over_8s": n_8s, "over_5s": n_5s, "under_1s": n_sh,
        "readability_score": score,
    }
    if job_log:
        job_log.info(
            f"[timeline_quality] score={score}/100 ({grade}) | "
            f"avg={avg_d:.1f}s over_8s={n_8s} over_5s={n_5s}",
            stage="align",
        )
    return metrics
