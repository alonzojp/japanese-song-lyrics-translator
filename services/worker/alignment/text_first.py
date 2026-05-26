"""
Text-first lyric line reconstruction.

Builds line timing from a global word list using competitive boundary matching
against official lyric text.  Ignores WhisperX segment boundaries entirely.

Architecture
------------
  - Single forward pointer through all time-sorted words (O(n) streaming)
  - Each word is evaluated against BOTH the current line and the next line
  - Ownership handed to the next line the moment the next line scores higher
    than the current line's continuation (+ sufficient current-line coverage)
  - No coverage-threshold exit: a line continues consuming words until the
    boundary competition resolves cleanly

Normalisation
-------------
  Japanese : NFKC, lowercase, strip punctuation / whitespace
             single-char kana/kanji tokens map 1-to-1 to lyric chars
  English  : same pipeline; space/hyphen collapse already handled by
             _STRIP_CHARS; single-char fragment tokens (T+r+u+e) match
             individual lyric chars naturally via prefix logic
"""
from __future__ import annotations

import logging
import unicodedata

logger = logging.getLogger(__name__)

# ── Timing nudges ──────────────────────────────────────────────────────────────
_PRELOAD_S = 0.10    # lead-in before first matched word's start
_TAIL_S    = 0.08    # tail after last matched word's end

# ── Competitive boundary parameters ───────────────────────────────────────────
# Fraction of the current line's normalised chars that must be matched before
# we are willing to hand a word off to the next line.
_MIN_HANDOFF_COV = 0.50

# ── Safety caps (prevent runaway scans) ───────────────────────────────────────
_MAX_CONSEC_SKIP = 8     # consecutive non-matching words before soft break
_MAX_SCAN        = 80    # total words examined per line (hard upper bound)
_SEARCH_WINDOW   = 40    # lyric-char look-ahead for near-match detection

# ── Minimum match coverage to produce a timed result (not a fallback) ─────────
_MIN_RESULT_COV  = 0.40

# ── Characters to strip when normalising ──────────────────────────────────────
# Listed as individual codepoints to avoid quoting/range ambiguities.
_STRIP_CHARS = frozenset([
    # ASCII whitespace and punctuation
    ' ', '\t', '\n', '\r', '!', '?', ',', '.', "'", '"', '-',
    # Ideographic space
    '　',
    # Japanese punctuation
    '、', '。',   # 、。
    '！', '？',   # ！？
    '，', '．',   # ，．
    '「', '」',   # 「」
    '『', '』',   # 『』
    '…', '・',   # …・
    '·',             # middle dot ·
    '（', '）',   # （）
    '［', '］',   # ［］
    '《', '》',   # 《》
    '〈', '〉',   # 〈〉
    'ー',             # ー  katakana-hiragana prolonged sound mark
    '―',             # ― horizontal bar
    # Unicode dashes
    '—', '–',   # — –
    # Unicode curly quotes
    chr(0x2018), chr(0x2019),  # left/right single quotation marks
    chr(0x201c), chr(0x201d),  # left/right double quotation marks
])


# ── Normalisation ──────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    """NFKC + lowercase + strip all punctuation and whitespace."""
    t = unicodedata.normalize('NFKC', text)
    t = t.lower()
    return ''.join(c for c in t if c not in _STRIP_CHARS)


# ── Core match primitive ───────────────────────────────────────────────────────

def _match_word(word_norm: str, remaining: str) -> tuple[float, int, int]:
    """
    Attempt to match word_norm at the head of remaining lyric text.

    Returns (quality, consume, skip):
      quality : match confidence [0.0–1.0]; 0.0 = no match
      consume : lyric chars consumed by this word (after skipping)
      skip    : lyric chars to skip before the consume

    Quality semantics (used for competitive boundary comparison):
      1.0        exact prefix match
      0.5–1.0    partial prefix (fraction of word matched)
      0.05–0.5   exact or partial match further inside the window
      0.0        no match found

    Search order:
      1. Exact prefix
      2. Partial prefix (≥50% of word length)
      3. Exact match within _SEARCH_WINDOW lyric chars (with skip)
      4. Partial match within window (with skip)
    """
    if not word_norm or not remaining:
        return 0.0, 0, 0

    n    = len(word_norm)
    half = max(1, (n + 1) // 2)

    # 1. Exact prefix
    if remaining.startswith(word_norm):
        return 1.0, n, 0

    # 2. Partial prefix (≥50% of word)
    for plen in range(n - 1, half - 1, -1):
        if remaining.startswith(word_norm[:plen]):
            return plen / n, plen, 0

    # 3. Exact match within search window (skip some lyric chars)
    window = remaining[:_SEARCH_WINDOW]
    idx = window.find(word_norm)
    if idx > 0:
        return max(0.05, 0.5 - 0.04 * idx), n, idx

    # 4. Partial match within search window
    for plen in range(n - 1, half - 1, -1):
        idx = window.find(word_norm[:plen])
        if idx > 0:
            return max(0.02, (plen / n) * 0.35 - 0.02 * idx), plen, idx

    return 0.0, 0, 0


# ── Competitive boundary matching ─────────────────────────────────────────────

def _match_line(
    lyric_norm: str,
    next_norm:  str,
    words:      list[dict],
    start_ptr:  int,
) -> tuple[list[dict], int, float]:
    """
    Advance the global word pointer, assigning words to the current line.

    At every word position, compare:
      cont_q — how well this word matches the current line's remaining text
      next_q — how well this word matches the NEXT line's opening text

    Handoff condition:
      next_q > cont_q  AND  coverage_so_far >= _MIN_HANDOFF_COV

    This ensures ownership passes to the next line the moment the next line
    makes a stronger claim, rather than stopping purely on coverage %.
    The _MIN_HANDOFF_COV guard prevents handing off before the current line
    has established enough territory (ambiguous lead-in tokens).

    Safety caps:
      - Emergency exit after _MAX_CONSEC_SKIP * 3 consecutive non-matching words
      - Hard scan cap of _MAX_SCAN total words examined

    Returns (matched, new_ptr, coverage).
    """
    total_len = len(lyric_norm)
    if total_len == 0:
        return [], start_ptr, 1.0

    remaining = lyric_norm
    matched:  list[dict] = []
    ptr       = start_ptr
    n         = len(words)
    consec    = 0
    scanned   = 0

    while remaining and ptr < n and scanned < _MAX_SCAN:
        w  = words[ptr]
        wn = _norm(w.get('word', ''))
        scanned += 1

        if not wn:
            ptr += 1
            continue

        cont_q, cont_c, cont_s = _match_word(wn, remaining)
        next_q, _,      _      = _match_word(wn, next_norm) if next_norm else (0.0, 0, 0)
        cov = 1.0 - len(remaining) / total_len

        # ── Competitive boundary: yield this word to the next line ─────────
        # next_q strictly greater — ties stay with the current line.
        if next_q > cont_q and cov >= _MIN_HANDOFF_COV:
            break

        # ── Match: consume ─────────────────────────────────────────────────
        if cont_q > 0.0:
            remaining = remaining[cont_s + cont_c:]
            matched.append(w)
            ptr   += 1
            consec = 0

        # ── No match: skip, check safety cap ──────────────────────────────
        else:
            ptr    += 1
            consec += 1
            if consec >= _MAX_CONSEC_SKIP * 3:   # 24 consecutive misses
                break

    coverage = 1.0 - len(remaining) / total_len
    return matched, ptr, coverage


# ── Public entry point ────────────────────────────────────────────────────────

def reconstruct_lines_text_first(
    official_lines: list[dict],
    all_words:      list[dict],
    job_log=None,
) -> list[dict]:
    """
    Rebuild lyric-line timing by assigning time-sorted words to official lines
    via competitive boundary matching.

    Inputs:
      official_lines — [{text, ...}] ordered ground-truth lyric lines
      all_words      — [{word, start, end, score, ...}] WhisperX words (any order)

    Returns segment-like dicts compatible with build_lyric_lines / visual_timing:
      [{text, start, end, words:[{word, start, end, score}], _timing_source}]

    WhisperX segment boundaries are ignored entirely.
    start/end derive from matched word timestamps ± _PRELOAD_S / _TAIL_S.
    Lines with insufficient word coverage receive interpolated fallback timing.
    """
    def _log(msg: str) -> None:
        if job_log:
            job_log.info(msg, stage='transcribe')
        else:
            logger.debug(msg)

    # Filter and sort all words by start time
    words = sorted(
        [w for w in all_words if w.get('word') and w.get('start') is not None],
        key=lambda w: float(w['start']),
    )
    n_words = len(words)

    # Pre-compute normalised lyric text for every line (used for next_norm lookup)
    norms = [_norm(ln.get('text', '')) for ln in official_lines]

    _log(f'[text_first] Reconstructing {len(official_lines)} lines from {n_words} words')

    result: list[dict] = []
    ptr = 0

    for line_idx, line in enumerate(official_lines):
        text = line.get('text', '').strip()
        if not text:
            continue

        lyric_norm = norms[line_idx]
        # Next non-empty line's normalised text (for competitive boundary)
        next_norm = ''
        for j in range(line_idx + 1, len(official_lines)):
            candidate = norms[j]
            if candidate:
                next_norm = candidate
                break

        matched, new_ptr, coverage = _match_line(lyric_norm, next_norm, words, ptr)

        _log(
            f'[text_first] L{line_idx:02d} cov={coverage:.0%} '
            f'words={len(matched)} ptr {ptr}->{new_ptr}  '
            f"'{text[:40]}'"
        )

        if matched and coverage >= _MIN_RESULT_COV:
            w_start = float(matched[0].get('start') or 0)
            w_end   = float(matched[-1].get('end') or matched[-1].get('start') or w_start)
            start   = max(0.0, round(w_start - _PRELOAD_S, 3))
            end     = round(w_end + _TAIL_S, 3)

            result.append({
                'text':           text,
                'start':          start,
                'end':            end,
                'words': [
                    {
                        'word':  w.get('word', ''),
                        'start': round(float(w.get('start') or 0), 3),
                        'end':   round(float(w.get('end') or w.get('start') or 0), 3),
                        'score': w.get('score', 1.0),
                    }
                    for w in matched
                ],
                '_timing_source': 'text_first',
            })
            ptr = new_ptr

        else:
            # No words matched (or coverage too low): interpolate from neighbours
            prev_end = result[-1]['end'] if result else 0.0
            if ptr < n_words:
                horizon = float(words[ptr].get('start') or prev_end + 3.0)
            else:
                horizon = prev_end + 3.0

            n_rem = max(1, sum(
                1 for lx in official_lines[line_idx:]
                if lx.get('text', '').strip()
            ))
            slot  = max(1.0, (horizon - prev_end) / n_rem)
            est_s = round(prev_end, 3)
            est_e = round(est_s + slot * 0.9, 3)

            _log(f'[text_first] L{line_idx:02d} NO MATCH — fallback {est_s:.2f}-{est_e:.2f}s')
            result.append({
                'text':           text,
                'start':          est_s,
                'end':            est_e,
                'words':          [],
                '_timing_source': 'text_first_fallback',
            })
            # Do not advance ptr — next line may find these words

    _log(
        f'[text_first] Done: {len(result)} lines, '
        f'{ptr}/{n_words} words consumed ({100 * ptr // max(1, n_words)}%)'
    )
    return result
