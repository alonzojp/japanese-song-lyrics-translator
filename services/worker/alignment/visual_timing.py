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
  - displayEnd is anchored to last aligned WORD boundary, not segment end.
    Silence, sustain, and segment tails after the last word are excluded.
  - Instrumental gaps are not absorbed into neighbouring lyric windows.
  - Mora-based duration compression prevents 15s display windows.
  - All logic runs offline. The playback layer reads times and does nothing else.
"""
from __future__ import annotations

import unicodedata

# ── Constants ─────────────────────────────────────────────────────────────────

_MORA_PER_SEC       = 0.22   # expected seconds per mora in sung Japanese
_VIS_MIN_S          = 1.2    # minimum display duration
_VIS_MAX_S          = 6.5    # maximum display duration
_TRANSITION_BIAS_S  = 0.15   # advance end earlier (early-transition UX)
_INST_GAP_THRESH_S  = 4.0    # gap > this = instrumental break
_VOCAL_GAP_THRESH_S = 1.5    # gap > this = notable vocal pause
_MAX_TAIL_INST_S    = 0.35   # max tail before instrumental gap
_MAX_TAIL_VOCAL_S   = 0.55   # max tail before vocal gap
_SMALL_KANA = frozenset('ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ')


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


def _compression_strength(ratio: float) -> float:
    """Acoustic-to-target ratio → compression strength [0, 1]."""
    if ratio <= 1.5:  return 0.0
    elif ratio <= 2.2: return _lerp(0.0,  0.30, (ratio - 1.5) / 0.7)
    elif ratio <= 3.0: return _lerp(0.30, 0.60, (ratio - 2.2) / 0.8)
    elif ratio <= 4.0: return _lerp(0.60, 0.80, (ratio - 3.0) / 1.0)
    elif ratio <= 6.0: return _lerp(0.80, 0.95, (ratio - 4.0) / 2.0)
    else:              return 0.95


def pick_best_anchor(
    line:      dict,
    next_line: dict | None = None,
) -> tuple[float, str]:
    """
    Choose the best end-anchor for a lyric line.

    Returns (anchor_end, source_label).

    Candidates (in preference order):
      1. last_word.end      — primary; excluded if unreliably early or overlapping
      2. second_last_word.end — fallback when last word is a short unstable token
      3. segment_end (acousticEnd) — last resort when word timestamps are absent/bad

    Scoring criteria (all deterministic):
      - Overlap penalty: candidate > next_line.startTime - 0.05  →  disqualified
      - Minimum coverage: candidate < a_start + VIS_MIN_S        →  disqualified
      - Recency penalty: large gap between candidate and acousticEnd = under-coverage
      - Stability bonus: second_last preferred over last when last is suspiciously
        short (< 150ms duration), which indicates a WhisperX boundary artifact
    """
    words    = line.get('words') or []
    a_start  = float(line.get('acousticStart', line.get('startTime', 0)))
    a_end    = float(line.get('acousticEnd',   line.get('endTime',   0)))
    next_start = float(next_line.get('startTime', a_end + 99.0)) if next_line else a_end + 99.0

    # Hard ceiling: anchor must not cross into next line
    overlap_ceiling = next_start - 0.05

    def valid(t: float) -> bool:
        return (
            t > a_start + _VIS_MIN_S * 0.5   # not absurdly early
            and t <= overlap_ceiling          # doesn't overlap next line
            and t <= a_end + 0.1              # doesn't exceed acoustic end by much
        )

    def score(t: float) -> float:
        """Higher = better. All terms are deterministic."""
        s = 0.0
        # Reward closeness to acousticEnd (up to 0 penalty at a_end)
        # Each 100ms away from a_end costs 1 point — prevents early cutoffs
        s -= abs(a_end - t) * 10.0
        # Reward being safely before next line
        margin = overlap_ceiling - t
        if margin < 0.3:
            s -= (0.3 - margin) * 20.0   # heavy penalty for tight margin
        return s

    # Build candidate list: (timestamp, label, bonus)
    # Bonus rewards word-based anchors over segment fallback
    candidates: list[tuple[float, str, float]] = []

    if len(words) >= 1:
        last_end = words[-1].get('end')
        if last_end is not None:
            last_end = float(last_end)
            # Detect unstable last token: very short duration (< 150ms)
            last_start = float(words[-1].get('start', last_end))
            last_dur   = last_end - last_start
            stability_bonus = 0.0 if last_dur >= 0.15 else -5.0
            if valid(last_end):
                candidates.append((last_end, 'last_word', 8.0 + stability_bonus))

    if len(words) >= 2:
        sl_end = words[-2].get('end')
        if sl_end is not None:
            sl_end = float(sl_end)
            if valid(sl_end):
                # Second-last is preferred only when last word is unstable —
                # give it a baseline bonus below stable last_word
                candidates.append((sl_end, 'second_last_word', 3.0))

    # Segment end is always a candidate (last resort)
    if valid(a_end):
        candidates.append((a_end, 'segment_end', 0.0))
    elif candidates:
        pass   # segment_end invalid (overlaps next line); word candidates cover it
    else:
        # Nothing valid — return segment end clamped to overlap ceiling
        return (min(a_end, overlap_ceiling), 'segment_end_clamped')

    if not candidates:
        return (min(a_end, overlap_ceiling), 'segment_end_clamped')

    # Pick highest (base_bonus + continuity_score)
    best = max(candidates, key=lambda c: c[2] + score(c[0]))
    return (best[0], best[1])


# ── Phase 1: visual normalization ─────────────────────────────────────────────

def visual_timing_normalization(lines: list[dict], job_log=None) -> list[dict]:
    """
    Overwrite startTime/endTime with display-optimised values.

    Acoustic originals preserved as acousticStart / acousticEnd.
    After this call, startTime/endTime reflect display timing only.
    finalize_timeline() must be called after this to enforce monotonicity.
    """
    n = len(lines)
    n_word_anchored = 0
    n_compressed    = 0
    n_sustain       = 0
    n_inst_capped   = 0
    total_saved     = 0.0

    for i, line in enumerate(lines):
        a_start = float(line.get('startTime', 0))
        a_end   = float(line.get('endTime',   0))
        a_dur   = max(0.0, a_end - a_start)

        # Preserve acoustic originals
        line['acousticStart'] = round(a_start, 3)
        line['acousticEnd']   = round(a_end,   3)

        # ── Timing anchor: scored multi-candidate selection ──────────────
        next_line  = lines[i + 1] if i + 1 < n else None
        anchor_end, anchor_src = pick_best_anchor(line, next_line)
        if 'word' in anchor_src:
            n_word_anchored += 1

        anchor_dur = max(0.0, anchor_end - a_start)

        # ── Mora-based visual target ─────────────────────────────────────
        moras      = _mora_count(line.get('text', ''))
        vis_target = max(_VIS_MIN_S, min(_VIS_MAX_S, moras * _MORA_PER_SEC))
        ratio      = anchor_dur / vis_target if vis_target > 0 else 1.0

        if ratio <= 1.1:
            # Within range — apply transition bias only
            display_end   = max(a_start + _VIS_MIN_S, anchor_end - _TRANSITION_BIAS_S)
            sustain_heavy = False
        else:
            strength       = _compression_strength(ratio)
            compressed_dur = _lerp(anchor_dur, vis_target, strength)
            sustain_heavy  = ratio > 4.0
            display_end    = a_start + compressed_dur - _TRANSITION_BIAS_S
            display_end    = min(display_end, anchor_end)  # never past word anchor

        # ── Gap-aware tail cap ──────────────────────────────────────────
        inst_capped = False
        gap_to_next = 0.0
        if i + 1 < n:
            nxt         = float(lines[i + 1].get('startTime', a_end))
            gap_to_next = nxt - a_end
            if gap_to_next > _INST_GAP_THRESH_S:
                cap = anchor_end + _MAX_TAIL_INST_S
                if display_end > cap:
                    display_end   = cap
                    inst_capped   = True
                    n_inst_capped += 1
            elif gap_to_next > _VOCAL_GAP_THRESH_S:
                cap = anchor_end + _MAX_TAIL_VOCAL_S
                display_end = min(display_end, cap)
        else:
            display_end = min(display_end, anchor_end + _MAX_TAIL_VOCAL_S)

        display_end = max(display_end, a_start + _VIS_MIN_S)

        saved = a_dur - (display_end - a_start)
        if saved > 0.1:
            n_compressed += 1
            total_saved  += saved
        if sustain_heavy:
            n_sustain += 1

        # Overwrite canonical times with display-optimised values
        line['startTime'] = round(a_start,   3)  # unchanged — start is acoustic
        line['endTime']   = round(display_end, 3)

        if job_log:
            flags = []
            if sustain_heavy:
                flags.append('sustain')
            if inst_capped:
                flags.append(f'inst-gap={gap_to_next:.1f}s')
            if anchor_src == 'word' and anchor_end < a_end - 0.5:
                flags.append(f'word-saved={(a_end - anchor_end):.1f}s')
            job_log.info(
                f"[visual_norm] L{i:02d}: "
                f"acoustic={a_dur:.1f}s anchor={anchor_src}({anchor_dur:.1f}s) "
                f"target={vis_target:.1f}s ratio={ratio:.2f} "
                f"→ {a_start:.3f}–{display_end:.3f}s saved={saved:.1f}s"
                + (f" [{', '.join(flags)}]" if flags else ""),
                stage="align",
            )

    if job_log:
        avg_s = total_saved / n_compressed if n_compressed else 0.0
        job_log.info(
            f"[visual_norm] summary: word_anchored={n_word_anchored}/{n} "
            f"compressed={n_compressed} sustain={n_sustain} "
            f"inst_capped={n_inst_capped} total_saved={total_saved:.1f}s "
            f"avg={avg_s:.1f}s/line",
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
    startTimes are NEVER modified — only endTimes are capped.
    """
    if not lines:
        return lines

    lines.sort(key=lambda l: l.get('startTime', 0))

    n_capped   = 0
    n_extended = 0

    for i, line in enumerate(lines):
        start = float(line['startTime'])
        end   = float(line['endTime'])

        # Enforce minimum duration
        if end < start + _VIS_MIN_S:
            new_end = round(start + _VIS_MIN_S, 3)
            if job_log and abs(new_end - end) > 0.01:
                job_log.info(
                    f"[finalize] L{i:02d}: extend {end:.3f} → {new_end:.3f}s "
                    f"(below min duration)",
                    stage="align",
                )
            line['endTime'] = new_end
            n_extended += 1
            end = new_end

        # Cap if overlapping next line
        if i + 1 < len(lines):
            next_start = float(lines[i + 1].get('startTime', end))
            if end > next_start - 0.05:
                capped = round(next_start - 0.05, 3)
                capped = max(capped, start + _VIS_MIN_S)
                if job_log and abs(capped - end) > 0.01:
                    job_log.info(
                        f"[finalize] L{i:02d}: cap {end:.3f} → {capped:.3f}s "
                        f"(overlaps L{i+1:02d} start={next_start:.3f}s)",
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
            f"capped={n_capped} extended={n_extended} monotonic={status}",
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
