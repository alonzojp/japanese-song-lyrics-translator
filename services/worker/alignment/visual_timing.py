"""
Visual timing normalization — decouples acoustic alignment from karaoke display timing.

Adds displayStart/displayEnd to each LyricLine after acoustic alignment is complete.
The karaoke player renders from displayStart/displayEnd, NOT startTime/endTime.

Design principles:
  1. Karaoke readability > acoustic purity
  2. Sustained vowels must NOT extend display duration linearly
  3. Instrumental gaps must NOT inflate neighboring lyric windows
  4. WhisperX timestamps are anchors — visual timing is a separate optimisation target
  5. displayEnd is anchored to the last aligned WORD boundary, not segment end —
     silence, sustain, and instrumental tails after the last word are excluded
"""
from __future__ import annotations

import unicodedata

# ── Constants ─────────────────────────────────────────────────────────────────

_MORA_PER_SEC          = 0.22   # expected seconds per mora in sung Japanese
_VIS_MIN_S             = 1.2    # floor: never display a line for less than this
_VIS_MAX_S             = 6.5    # ceiling: maximum karaoke display window
_TRANSITION_BIAS_S     = 0.15   # advance display_end earlier (early-transition UX)
_INST_GAP_THRESH_S     = 4.0    # gap > this = instrumental break
_VOCAL_GAP_THRESH_S    = 1.5    # gap > this = notable vocal pause
_MAX_TAIL_INST_S       = 0.35   # max tail extension before an instrumental gap
_MAX_TAIL_VOCAL_S      = 0.55   # max tail extension before a vocal gap
_SMALL_KANA = frozenset('ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ')


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mora_count(text: str) -> int:
    """Estimate Japanese mora count (primary timing unit in singing)."""
    count = 0
    for ch in unicodedata.normalize('NFKC', text):
        cp = ord(ch)
        if ch in _SMALL_KANA:
            pass  # modifies previous mora, doesn't add one
        elif 0x3041 <= cp <= 0x3096 or 0x30A1 <= cp <= 0x30F6:  # hiragana / katakana
            count += 1
        elif ch in ('ー', '〜'):  # long vowel marker
            count += 1
        elif 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:  # kanji
            count += 2
        elif ch.isascii() and ch.isalpha():
            count += 1
    return max(1, count)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def _compression_strength(ratio: float) -> float:
    """
    Map acoustic_ratio (acoustic_dur / vis_target) → compression strength [0, 1].

    ratio ≤ 1.5  → no compression    (acoustic is already good)
    ratio ~2.2   → gentle (30%)
    ratio ~3.0   → strong (60%)
    ratio ~4.0   → very strong (80%)
    ratio  5.0+  → near-maximum (95%)
    """
    if ratio <= 1.5:
        return 0.0
    elif ratio <= 2.2:
        return _lerp(0.0,  0.30, (ratio - 1.5) / 0.7)
    elif ratio <= 3.0:
        return _lerp(0.30, 0.60, (ratio - 2.2) / 0.8)
    elif ratio <= 4.0:
        return _lerp(0.60, 0.80, (ratio - 3.0) / 1.0)
    elif ratio <= 6.0:
        return _lerp(0.80, 0.95, (ratio - 4.0) / 2.0)
    else:
        return 0.95


def _last_word_end(line: dict) -> float | None:
    """Return the end timestamp of the last aligned word, or None."""
    words = line.get('words')
    if not words:
        return None
    last = words[-1]
    end  = last.get('end')
    if end is None or end <= 0:
        return None
    return float(end)


def _first_word_start(line: dict) -> float | None:
    """Return the start timestamp of the first aligned word, or None."""
    words = line.get('words')
    if not words:
        return None
    first = words[0]
    start = first.get('start')
    if start is None or start < 0:
        return None
    return float(start)


# ── Main entry point ──────────────────────────────────────────────────────────

def visual_timing_normalization(lines: list[dict], job_log=None) -> list[dict]:
    """
    Add displayStart/displayEnd to each LyricLine dict.

    Must be called after build_lyric_lines() and before save_aligned_lines().
    Mutates lines in-place and returns the list.

    Timing source hierarchy:
      1. Last aligned word boundary  — best anchor (excludes silence/sustain after last word)
      2. Mora-compressed segment end — used when no word timestamps available
      3. Proportional minimum        — last resort (vis_min floor)

    displayStart = acoustic_start (same as startTime — we don't shift line starts)
    displayEnd   = word_end (or compressed acoustic_end), minus transition bias,
                   subject to instrumental-gap and next-line caps
    """
    n = len(lines)

    n_word_anchored = 0
    n_compressed    = 0
    n_sustain       = 0
    n_inst_capped   = 0
    total_saved     = 0.0

    for i, line in enumerate(lines):
        acoustic_start = float(line.get('startTime', 0))
        acoustic_end   = float(line.get('endTime',   0))
        acoustic_dur   = max(0.0, acoustic_end - acoustic_start)

        # ── Timing anchor: prefer last word boundary ─────────────────────────
        last_word = _last_word_end(line)
        # Only use last_word if it's within the segment window and not trivially 0
        if last_word is not None and acoustic_start <= last_word <= acoustic_end:
            anchor_end  = last_word
            anchor_src  = 'word_boundary'
            n_word_anchored += 1
        else:
            anchor_end  = acoustic_end
            anchor_src  = 'segment_end'

        anchor_dur = max(0.0, anchor_end - acoustic_start)

        # ── Visual target from mora count ────────────────────────────────────
        text       = line.get('text', '')
        moras      = _mora_count(text)
        vis_target = max(_VIS_MIN_S, min(_VIS_MAX_S, moras * _MORA_PER_SEC))

        # ── Compute display end ──────────────────────────────────────────────
        anchor_ratio = anchor_dur / vis_target if vis_target > 0 else 1.0

        if anchor_ratio <= 1.1:
            # Anchor is already within target range — just apply bias
            display_end = max(
                acoustic_start + _VIS_MIN_S,
                anchor_end - _TRANSITION_BIAS_S,
            )
            compression  = 0.0
            sustain_heavy = False
        else:
            # Over-expanded — compress toward visual target
            strength       = _compression_strength(anchor_ratio)
            compressed_dur = _lerp(anchor_dur, vis_target, strength)
            sustain_heavy  = anchor_ratio > 4.0
            display_end    = acoustic_start + compressed_dur - _TRANSITION_BIAS_S
            # Never exceed the word anchor (sustain / silence after last word excluded)
            display_end    = min(display_end, anchor_end)
            compression    = strength

        # ── Gap-aware tail capping ────────────────────────────────────────────
        inst_capped = False
        gap_to_next = 0.0
        if i + 1 < n:
            next_start  = float(lines[i + 1].get('startTime', acoustic_end))
            gap_to_next = next_start - acoustic_end
            if gap_to_next > _INST_GAP_THRESH_S:
                # Instrumental break — hard cap
                cap = anchor_end + _MAX_TAIL_INST_S
                if display_end > cap:
                    display_end   = cap
                    inst_capped   = True
                    n_inst_capped += 1
            elif gap_to_next > _VOCAL_GAP_THRESH_S:
                # Vocal gap — moderate cap
                cap = anchor_end + _MAX_TAIL_VOCAL_S
                if display_end > cap:
                    display_end = cap
        else:
            display_end = min(display_end, anchor_end + _MAX_TAIL_VOCAL_S)

        # Enforce minimum display duration
        display_end = max(display_end, acoustic_start + _VIS_MIN_S)

        # ── Track metrics ─────────────────────────────────────────────────────
        display_dur = display_end - acoustic_start
        saved = acoustic_dur - display_dur
        if saved > 0.1:
            n_compressed += 1
            total_saved  += saved
        if sustain_heavy:
            n_sustain += 1

        line['displayStart'] = round(acoustic_start, 3)
        line['displayEnd']   = round(display_end,    3)

        if job_log:
            flags = []
            if sustain_heavy:
                flags.append('sustain-heavy')
            if inst_capped:
                flags.append(f'inst-gap={gap_to_next:.1f}s')
            if anchor_src == 'word_boundary' and anchor_end < acoustic_end - 0.5:
                flags.append(f'word-anchor-saved={(acoustic_end - anchor_end):.1f}s')
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            job_log.info(
                f"[visual_norm] L{i:02d}: "
                f"acoustic={acoustic_dur:.1f}s anchor={anchor_src}({anchor_dur:.1f}s) "
                f"target={vis_target:.1f}s ratio={anchor_ratio:.2f} "
                f"display={acoustic_start:.3f}–{display_end:.3f}s "
                f"saved={saved:.1f}s"
                + flag_str,
                stage="align",
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    if job_log:
        avg_saved = total_saved / n_compressed if n_compressed else 0.0
        job_log.info(
            f"[visual_norm] summary: {n} lines | "
            f"word_anchored={n_word_anchored} compressed={n_compressed} "
            f"sustain={n_sustain} inst_capped={n_inst_capped} "
            f"total_saved={total_saved:.1f}s avg_saved={avg_saved:.1f}s/line",
            stage="align",
        )

    return lines


# ── Quality metrics ───────────────────────────────────────────────────────────

def visual_timing_quality(lines: list[dict], job_log=None) -> dict:
    """
    Compute visual timing quality metrics after normalization.
    Emits a [visual_quality] log line.
    """
    if not lines:
        return {}

    display_durs = [
        max(0.0, l.get('displayEnd', l.get('endTime', 0))
                 - l.get('displayStart', l.get('startTime', 0)))
        for l in lines
    ]
    acoustic_durs = [
        max(0.0, l.get('endTime', 0) - l.get('startTime', 0))
        for l in lines
    ]

    n              = len(lines)
    n_over_8s      = sum(1 for d in display_durs if d > 8.0)
    n_over_5s      = sum(1 for d in display_durs if d > 5.0)
    n_short        = sum(1 for d in display_durs if d < 1.2)
    avg_display    = sum(display_durs) / n
    avg_acoustic   = sum(acoustic_durs) / n
    avg_vis_ratio  = avg_acoustic / avg_display if avg_display > 0 else 1.0

    score = 100
    score -= n_over_8s * 15
    score -= n_over_5s * 3
    score -= n_short   * 5
    score  = max(0, score)

    metrics = {
        "n_lines":           n,
        "avg_display_s":     round(avg_display,  2),
        "avg_acoustic_s":    round(avg_acoustic, 2),
        "avg_vis_ratio":     round(avg_vis_ratio, 2),
        "overlong_count":    n_over_8s,
        "long_count":        n_over_5s,
        "short_count":       n_short,
        "readability_score": score,
    }

    if job_log:
        grade = (
            "EXCELLENT" if score >= 90 else
            "GOOD"      if score >= 75 else
            "FAIR"      if score >= 55 else
            "POOR"
        )
        job_log.info(
            f"[visual_quality] score={score}/100 ({grade}) | "
            f"avg_display={avg_display:.1f}s avg_acoustic={avg_acoustic:.1f}s "
            f"vis_ratio={avg_vis_ratio:.2f} "
            f"overlong(>8s)={n_over_8s} long(>5s)={n_over_5s}",
            stage="align",
        )

    return metrics
