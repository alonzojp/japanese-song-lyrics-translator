"""
Lyrics preprocessor.

Takes a raw list of LyricLine objects and produces a clean, standardized list:
  - Strips noise lines
  - Merges very short continuation lines
  - Detects and flattens repeated choruses (marks them, optionally deduplicates)
  - Assigns confidence adjustments based on line quality
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Optional

from lyrics.types import LyricLine
from lyrics.normalizer import is_noise_line, normalize_line, japanese_char_ratio

# ── Config ─────────────────────────────────────────────────────────────────────

MIN_LINE_CHARS    = 2      # lines shorter than this are merged or dropped
MAX_LINE_CHARS    = 120    # lines longer than this may be split
CHORUS_SIM_THRESH = 0.82   # SequenceMatcher ratio to consider two blocks "same"
MIN_CHORUS_LINES  = 2      # minimum lines in a block to consider it a chorus


# ── Public API ─────────────────────────────────────────────────────────────────

def preprocess(
    lines:            list[LyricLine],
    *,
    deduplicate_chorus: bool = False,
) -> tuple[list[LyricLine], dict]:
    """
    Clean and standardize a list of LyricLine objects.

    Returns (processed_lines, stats_dict) where stats_dict contains:
      - original_count
      - final_count
      - dropped_noise
      - merged_short
      - chorus_groups (list of repeated block indices)
      - avg_japanese_ratio
    """
    stats: dict = {
        "original_count":  len(lines),
        "dropped_noise":   0,
        "merged_short":    0,
        "chorus_groups":   [],
        "avg_japanese_ratio": 0.0,
    }

    # 1. Normalize + drop noise
    clean: list[LyricLine] = []
    for ln in lines:
        norm = normalize_line(ln.text)
        if is_noise_line(norm):
            stats["dropped_noise"] += 1
            continue
        clean.append(LyricLine(
            text=norm,
            start_time=ln.start_time,
            end_time=ln.end_time,
        ))

    # 2. Merge continuation lines (very short lines glued to previous)
    clean, merged = _merge_short_lines(clean)
    stats["merged_short"] = merged

    # 3. Chorus detection
    chorus_groups = _detect_repeated_blocks(clean)
    stats["chorus_groups"] = [list(g) for g in chorus_groups]

    if deduplicate_chorus and chorus_groups:
        clean = _deduplicate_choruses(clean, chorus_groups)

    # 4. Compute stats
    ja_ratios = [japanese_char_ratio(ln.text) for ln in clean]
    stats["avg_japanese_ratio"] = (
        round(sum(ja_ratios) / len(ja_ratios), 3) if ja_ratios else 0.0
    )
    stats["final_count"] = len(clean)

    return clean, stats


def compute_confidence_adjustment(stats: dict) -> float:
    """
    Return a multiplier (0.5–1.0) to apply to a provider's base confidence
    based on preprocessing statistics.
    """
    adj = 1.0

    # Very few lines → probably wrong / instrumental
    count = stats.get("final_count", 0)
    if count < 4:
        adj *= 0.5
    elif count < 10:
        adj *= 0.8

    # Low Japanese character ratio → wrong language / corrupted
    ratio = stats.get("avg_japanese_ratio", 1.0)
    if ratio < 0.3:
        adj *= 0.5
    elif ratio < 0.6:
        adj *= 0.75

    return max(0.1, min(1.0, adj))


# ── Private helpers ────────────────────────────────────────────────────────────

def _merge_short_lines(lines: list[LyricLine]) -> tuple[list[LyricLine], int]:
    """Merge lines shorter than MIN_LINE_CHARS into the previous line."""
    if not lines:
        return lines, 0

    merged_count = 0
    result: list[LyricLine] = [lines[0]]

    for ln in lines[1:]:
        prev = result[-1]
        if len(ln.text) < MIN_LINE_CHARS:
            # Glue to previous
            result[-1] = LyricLine(
                text=prev.text + ln.text,
                start_time=prev.start_time,
                end_time=ln.end_time or prev.end_time,
            )
            merged_count += 1
        else:
            result.append(ln)

    return result, merged_count


def _block_key(lines: list[LyricLine], start: int, length: int) -> str:
    """Concatenate a block of lines for comparison."""
    return "\n".join(ln.text for ln in lines[start : start + length])


def _detect_repeated_blocks(lines: list[LyricLine]) -> list[frozenset[int]]:
    """
    Find groups of line-index-ranges that are near-duplicates (choruses).
    Returns a list of frozensets, each containing the start indices of matching blocks.
    """
    if len(lines) < MIN_CHORUS_LINES * 2:
        return []

    # Try block sizes from MIN_CHORUS_LINES up to len/3
    max_block = max(MIN_CHORUS_LINES, len(lines) // 3)
    groups: list[frozenset[int]] = []
    seen_starts: set[int] = set()

    for block_len in range(MIN_CHORUS_LINES, max_block + 1):
        for i in range(len(lines) - block_len):
            if i in seen_starts:
                continue
            key_i = _block_key(lines, i, block_len)
            matches = {i}
            for j in range(i + block_len, len(lines) - block_len + 1):
                if j in seen_starts:
                    continue
                key_j = _block_key(lines, j, block_len)
                ratio = SequenceMatcher(None, key_i, key_j).ratio()
                if ratio >= CHORUS_SIM_THRESH:
                    matches.add(j)

            if len(matches) >= 2:
                groups.append(frozenset(matches))
                seen_starts.update(matches)

    # Deduplicate subset groups
    unique: list[frozenset[int]] = []
    for g in groups:
        if not any(g < other for other in groups):
            unique.append(g)
    return unique


def _deduplicate_choruses(
    lines: list[LyricLine],
    chorus_groups: list[frozenset[int]],
) -> list[LyricLine]:
    """Keep only the first occurrence of each repeated block."""
    # Collect all non-first-occurrence start indices to remove
    remove_starts: set[int] = set()
    for group in chorus_groups:
        sorted_starts = sorted(group)
        for repeat_start in sorted_starts[1:]:  # keep first, remove rest
            remove_starts.add(repeat_start)

    # Determine which individual line indices to drop
    drop_indices: set[int] = set()
    for start in remove_starts:
        for k in range(MIN_CHORUS_LINES):
            drop_indices.add(start + k)

    return [ln for i, ln in enumerate(lines) if i not in drop_indices]
