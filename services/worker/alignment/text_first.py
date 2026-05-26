"""
Text-first lyric line reconstruction.

Builds line timing from a global word list using greedy sequential matching
against official lyric text. Ignores WhisperX segment boundaries entirely.

Implements STEP 2-4 of the text-first alignment spec:
  - Single forward pointer through all aligned words
  - Japanese: NFKC-normalised char/mora matching with partial-token tolerance
  - English: case-normalised, fragmentation-tolerant
  - 80% coverage declares a line complete; +/-2s window force-includes
    adjacent words that are acoustically part of the span
  - <50% coverage -> time-interpolated fallback, no words
"""
from __future__ import annotations

import logging
import unicodedata

logger = logging.getLogger(__name__)

# Timing nudges (spec STEP 3)
_PRELOAD_S = 0.10    # lead-in before first matched word's start
_TAIL_S    = 0.08    # tail after last matched word's end

# Coverage thresholds (spec STEP 4)
_ACCEPT_COVERAGE = 0.80
_REJECT_COVERAGE = 0.50

# Scan limits
_MAX_CONSEC_SKIP = 8
_MAX_SCAN        = 60
_SEARCH_WINDOW   = 40   # chars of remaining lyric to search within

# Force-include window (spec STEP 2 robust fallback)
_NEAR_WINDOW_S = 2.0

# Characters to strip when normalising text for matching.
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
    # Unicode curly single/double quotes (chr avoids embedding non-ASCII in source)
    chr(0x2018), chr(0x2019),  # left/right single quotation marks
    chr(0x201c), chr(0x201d),  # left/right double quotation marks
])


def _norm(text: str) -> str:
    """NFKC + lowercase + strip all punctuation and whitespace."""
    t = unicodedata.normalize('NFKC', text)
    t = t.lower()
    t = ''.join(c for c in t if c not in _STRIP_CHARS)
    return t


def _find_match(word_norm: str, remaining: str) -> tuple[int, int]:
    """
    Locate word_norm in the head of remaining.

    Returns (consume, skip):
      consume > 0  -> chars to remove from remaining (the matched portion)
      skip         -> lyric chars to skip before the match
      consume == 0 -> no match found

    Search order:
      1. Exact prefix
      2. Partial prefix (>=50% of word)
      3. Exact match within first SEARCH_WINDOW lyric chars
      4. Partial match within search window
    """
    if not word_norm or not remaining:
        return 0, 0

    # 1. Exact prefix
    if remaining.startswith(word_norm):
        return len(word_norm), 0

    # 2. Partial prefix (>=50% of word length, minimum 1 char)
    half = max(1, (len(word_norm) + 1) // 2)
    for plen in range(len(word_norm) - 1, half - 1, -1):
        if remaining.startswith(word_norm[:plen]):
            return plen, 0

    # 3. Exact match within search window
    window = remaining[:_SEARCH_WINDOW]
    idx = window.find(word_norm)
    if idx > 0:
        return len(word_norm), idx

    # 4. Partial match within search window
    for plen in range(len(word_norm) - 1, half - 1, -1):
        idx = window.find(word_norm[:plen])
        if idx > 0:
            return plen, idx

    return 0, 0


def _greedy_match_line(
    lyric_text: str,
    words:      list[dict],
    start_ptr:  int,
) -> tuple[list[dict], int, float]:
    """
    Advance the global word pointer, consuming words that match lyric_text.

    Returns (matched, new_ptr, coverage):
      matched   - words assigned to this line
      new_ptr   - updated global pointer
      coverage  - fraction of normalised lyric chars accounted for
    """
    lyric_norm = _norm(lyric_text)
    total_len  = len(lyric_norm)
    if total_len == 0:
        return [], start_ptr, 1.0

    remaining:    str        = lyric_norm
    matched:      list[dict] = []
    ptr:          int        = start_ptr
    n             = len(words)
    consec_skip   = 0
    total_scanned = 0

    while remaining and ptr < n and total_scanned < _MAX_SCAN:
        w  = words[ptr]
        wn = _norm(w.get('word', ''))
        total_scanned += 1

        if not wn:
            ptr += 1
            continue

        consume, skip = _find_match(wn, remaining)

        if consume > 0:
            remaining = remaining[skip + consume:]
            matched.append(w)
            ptr += 1
            consec_skip = 0
        else:
            # Word does not match remaining lyric text:
            # could be an ASR error token or belong to the next line.
            ptr += 1
            consec_skip += 1

            coverage_now = 1.0 - len(remaining) / total_len
            if consec_skip >= _MAX_CONSEC_SKIP:
                if coverage_now >= _ACCEPT_COVERAGE:
                    break
                elif coverage_now >= _REJECT_COVERAGE and consec_skip >= _MAX_CONSEC_SKIP * 2:
                    break
                elif consec_skip >= _MAX_CONSEC_SKIP * 3:
                    break

    coverage = 1.0 - len(remaining) / total_len
    return matched, ptr, coverage


def _force_include_nearby(
    matched: list[dict],
    words:   list[dict],
    ptr:     int,
) -> tuple[list[dict], int]:
    """
    After >=80% coverage: absorb consecutively adjacent words within
    _NEAR_WINDOW_S seconds of the matched span's end.

    Implements spec STEP 2 robust fallback rule.
    """
    if not matched:
        return matched, ptr

    span_end  = float(matched[-1].get('end') or matched[-1].get('start') or 0)
    extra:     list[dict] = []
    p         = ptr

    while p < len(words):
        w  = words[p]
        ws = float(w.get('start') or 0)

        if ws > span_end + _NEAR_WINDOW_S:
            break

        if ws >= span_end:
            prev = (extra[-1] if extra else matched[-1])
            prev_end = float(prev.get('end') or prev.get('start') or ws)
            if ws - prev_end > _NEAR_WINDOW_S:
                break
            extra.append(w)
        p += 1

    return matched + extra, p


def reconstruct_lines_text_first(
    official_lines: list[dict],
    all_words:      list[dict],
    job_log=None,
) -> list[dict]:
    """
    Rebuild lyric-line timing by greedily assigning global words to official lines.

    Inputs:
      official_lines - [{text, ...}] ordered ground-truth lyric lines
      all_words      - [{word, start, end, ...}] WhisperX words, any order

    Returns segment-like dicts compatible with build_lyric_lines:
      [{text, start, end, words, _timing_source}]

    start/end are set from word timestamps (with _PRELOAD_S / _TAIL_S nudges).
    Segment boundaries from WhisperX are completely ignored.
    """
    def _log(msg: str) -> None:
        if job_log:
            job_log.info(msg, stage='transcribe')
        else:
            logger.debug(msg)

    words = sorted(
        [w for w in all_words if w.get('word') and w.get('start') is not None],
        key=lambda w: float(w['start']),
    )
    n_words = len(words)

    _log(f'[text_first] Reconstructing {len(official_lines)} lines from {n_words} words')

    result: list[dict] = []
    ptr = 0

    for line_idx, line in enumerate(official_lines):
        text = line.get('text', '').strip()
        if not text:
            continue

        matched, new_ptr, coverage = _greedy_match_line(text, words, ptr)

        # Force-include acoustically adjacent words only when there is still
        # unmatched lyric text (coverage < 1.0).  At full coverage the line is
        # complete and we must not absorb the next line's first word.
        if _ACCEPT_COVERAGE <= coverage < 1.0 and matched:
            matched, new_ptr = _force_include_nearby(matched, words, new_ptr)

        _log(
            f'[text_first] L{line_idx:02d} cov={coverage:.0%} '
            f'words={len(matched)} ptr {ptr}->{new_ptr}  '
            f"'{text[:40]}'"
        )

        if matched:
            w_start = float(matched[0].get('start') or 0)
            w_end   = float(
                matched[-1].get('end') or matched[-1].get('start') or w_start
            )
            start = max(0.0, round(w_start - _PRELOAD_S, 3))
            end   = round(w_end + _TAIL_S, 3)

            result.append({
                'text':           text,
                'start':          start,
                'end':            end,
                'words':          [
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
            # No words matched at all: interpolate from neighbours
            prev_end = result[-1]['end'] if result else 0.0
            if ptr < n_words:
                horizon = float(words[ptr].get('start') or prev_end + 3.0)
            else:
                horizon = prev_end + 3.0

            n_rem  = max(1, sum(
                1 for l in official_lines[line_idx:]
                if l.get('text', '').strip()
            ))
            slot  = max(1.0, (horizon - prev_end) / n_rem)
            est_s = round(prev_end, 3)
            est_e = round(est_s + slot * 0.9, 3)

            _log(
                f'[text_first] L{line_idx:02d} NO MATCH — '
                f'fallback {est_s:.2f}-{est_e:.2f}s'
            )
            result.append({
                'text':           text,
                'start':          est_s,
                'end':            est_e,
                'words':          [],
                '_timing_source': 'text_first_fallback',
            })
            # Do not advance ptr: next line can attempt from same position

    _log(
        f'[text_first] Done: {len(result)} lines, '
        f'{ptr}/{n_words} words consumed ({100*ptr//max(1,n_words)}%)'
    )
    return result
