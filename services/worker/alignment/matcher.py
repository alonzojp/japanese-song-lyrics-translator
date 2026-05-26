"""
Offline lyrics-to-transcript alignment using DTW + fuzzy matching.

Mirrors the TypeScript implementation in packages/alignment/src/.
Takes:
  transcript_segments  list[dict]  — from aligned_words.json
  official_lines       list[dict]  — from lyrics_cached.json
Returns:
  matched_lyrics.json  with AlignedLine objects

This runs as an optional step in the pipeline after step 4 (align),
when both a transcript and official lyrics are available for a video.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_DTW_WINDOW      = 0      # 0 = auto
DEFAULT_MIN_SIM         = 0.20
DEFAULT_ANCHOR_THRESH   = 0.62
DEFAULT_CHORUS_THRESH   = 0.78
DEFAULT_MIN_CHORUS_LINES = 2
MAX_HARD_DRIFT          = 2.5
LONG_SONG_THRESHOLD     = 40

# ── Normalization ──────────────────────────────────────────────────────────────

_KATA_TO_HIRA_OFFSET = 0x60
_PUNCT_RE            = re.compile(r'[。、！？!?.,…「」『』【】〈〉《》・\-\s　]+')
_PROLONGED_RE        = re.compile(r'ー+')
_REPEATED_RE         = re.compile(r'(.)\1{2,}')

# Minimal romaji table (hiragana only; katakana converted first)
_ROMAJI_MAP: dict[str, str] = {
    'あ':'a','い':'i','う':'u','え':'e','お':'o',
    'か':'ka','き':'ki','く':'ku','け':'ke','こ':'ko',
    'さ':'sa','し':'shi','す':'su','せ':'se','そ':'so',
    'た':'ta','ち':'chi','つ':'tsu','て':'te','と':'to',
    'な':'na','に':'ni','ぬ':'nu','ね':'ne','の':'no',
    'は':'ha','ひ':'hi','ふ':'fu','へ':'he','ほ':'ho',
    'ま':'ma','み':'mi','む':'mu','め':'me','も':'mo',
    'や':'ya','ゆ':'yu','よ':'yo',
    'ら':'ra','り':'ri','る':'ru','れ':'re','ろ':'ro',
    'わ':'wa','を':'wo','ん':'n',
    'が':'ga','ぎ':'gi','ぐ':'gu','げ':'ge','ご':'go',
    'ざ':'za','じ':'ji','ず':'zu','ぜ':'ze','ぞ':'zo',
    'だ':'da','ぢ':'ji','づ':'zu','で':'de','ど':'do',
    'ば':'ba','び':'bi','ぶ':'bu','べ':'be','ぼ':'bo',
    'ぱ':'pa','ぴ':'pi','ぷ':'pu','ぺ':'pe','ぽ':'po',
}


def _kata_to_hira(s: str) -> str:
    return ''.join(
        chr(ord(c) - _KATA_TO_HIRA_OFFSET) if 'ァ' <= c <= 'ヶ' else c
        for c in s
    )


def _to_romaji(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        two = s[i:i+2]
        if two in _ROMAJI_MAP:
            out.append(_ROMAJI_MAP[two])
            i += 2
        elif s[i] in _ROMAJI_MAP:
            out.append(_ROMAJI_MAP[s[i]])
            i += 1
        else:
            out.append(s[i])
            i += 1
    return ''.join(out)


def normalize(text: str) -> str:
    s = unicodedata.normalize('NFKC', text)
    s = _kata_to_hira(s)
    s = _PUNCT_RE.sub('', s)
    s = _PROLONGED_RE.sub('', s)
    s = _REPEATED_RE.sub(r'\1\1', s)
    return s.lower().strip()


def to_romaji(text: str) -> str:
    return re.sub(r'[^a-z0-9]', '', _to_romaji(_kata_to_hira(text)).lower())


# ── Similarity ─────────────────────────────────────────────────────────────────

def levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost)
        prev, curr = curr, prev
    return prev[n]


def lev_sim(a: str, b: str) -> float:
    if not a and not b: return 1.0
    mx = max(len(a), len(b))
    return 1.0 - levenshtein(a, b) / mx if mx else 1.0


def bigram_sim(a: str, b: str) -> float:
    if not a or not b: return 0.0
    if len(a) < 2 or len(b) < 2: return 1.0 if a == b else 0.0
    def bigrams(s: str) -> dict[str, int]:
        d: dict[str, int] = {}
        for i in range(len(s) - 1):
            bg = s[i:i+2]
            d[bg] = d.get(bg, 0) + 1
        return d
    ba, bb = bigrams(a), bigrams(b)
    inter = sum(min(ba.get(k, 0), bb.get(k, 0)) for k in ba)
    total = sum(ba.values()) + sum(bb.values())
    return 2 * inter / total if total else 0.0


def compute_sim(na: str, nb: str, ra: str, rb: str) -> float:
    return 0.50 * lev_sim(na, nb) + 0.30 * bigram_sim(na, nb) + 0.20 * lev_sim(ra, rb)


# ── DTW ────────────────────────────────────────────────────────────────────────

def dtw(cost: list[list[float]], window: int = 0) -> tuple[float, list[tuple[int,int]]]:
    n, m = len(cost), len(cost[0]) if cost else 0
    if n == 0 or m == 0:
        return 0.0, []

    eff_w = max(window, abs(n - m)) if window else max(n, m)
    INF   = float('inf')
    acc   = [[INF] * m for _ in range(n)]
    acc[0][0] = cost[0][0]

    for i in range(1, n):
        if abs(i) <= eff_w:
            acc[i][0] = cost[i][0] + acc[i-1][0]
    for j in range(1, m):
        if abs(j) <= eff_w:
            acc[0][j] = cost[0][j] + acc[0][j-1]

    for i in range(1, n):
        lo = max(1, i - eff_w) if window else 1
        hi = min(m, i + eff_w + 1) if window else m
        for j in range(lo, hi):
            acc[i][j] = cost[i][j] + min(acc[i-1][j], acc[i][j-1], acc[i-1][j-1])

    # Traceback
    path: list[tuple[int, int]] = []
    i, j = n - 1, m - 1
    path.append((i, j))
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            best = min(acc[i-1][j-1], acc[i-1][j], acc[i][j-1])
            if best == acc[i-1][j-1]:
                i -= 1; j -= 1
            elif best == acc[i-1][j]:
                i -= 1
            else:
                j -= 1
        path.append((i, j))
    path.reverse()
    return acc[n-1][m-1], path


def resolve_path(path: list[tuple[int,int]], n: int, cost: list[list[float]]) -> list[int]:
    from collections import defaultdict
    cands: dict[int, list[int]] = defaultdict(list)
    for i, j in path:
        cands[i].append(j)
    mapping = [-1] * n
    for i in range(n):
        js = cands.get(i, [])
        if js:
            mapping[i] = min(js, key=lambda j: cost[i][j] if j < len(cost[i]) else 1.0)
    return mapping


# ── Chorus detection ───────────────────────────────────────────────────────────

def detect_choruses(
    norm_lines:  list[str],
    threshold:   float = DEFAULT_CHORUS_THRESH,
    min_lines:   int   = DEFAULT_MIN_CHORUS_LINES,
) -> list[dict]:
    n = len(norm_lines)
    if n < min_lines * 2:
        return []

    groups: list[dict] = []
    seen: set[int] = set()

    for offset in range(min_lines, n):
        run_start = -1
        run_sim   = 0.0
        run_len   = 0

        for i in range(n - offset):
            j   = i + offset
            sim = bigram_sim(norm_lines[i], norm_lines[j])
            if sim >= threshold:
                if run_start == -1:
                    run_start, run_sim, run_len = i, 0.0, 0
                run_sim += sim
                run_len += 1
            else:
                if run_start != -1 and run_len >= min_lines and run_start not in seen:
                    groups.append({
                        'id':            len(groups),
                        'lineIndices':   list(range(run_start, run_start + run_len)),
                        'occurrences':   2,
                        'firstLineIndex': run_start,
                        'groupSize':     run_len,
                        'avgSimilarity': run_sim / run_len,
                    })
                    seen.update(range(run_start, run_start + run_len))
                run_start = -1

    return groups


# ── Main matcher ───────────────────────────────────────────────────────────────

@dataclass
class AlignedLine:
    id:              str
    text:            str
    transcript_text: str
    start_time:      float
    end_time:        float
    words:           list = field(default_factory=list)
    confidence:      float = 0.0
    text_similarity: float = 0.0
    temporal_score:  float = 0.0
    neighbor_score:  float = 0.0
    transcript_index: int  = -1
    is_chorus:       bool  = False
    chorus_group_id: Optional[int] = None
    drift:           float = 0.0
    anchor_distance: int   = -1


def _subdivide_shared_segments(lines: list[dict]) -> None:
    """
    Fallback: when multiple consecutive lines share the same segment,
    divide the time range evenly. Used only when char-level matching fails.
    """
    i = 0
    while i < len(lines):
        j = i + 1
        while j < len(lines) and lines[j]['startTime'] == lines[i]['startTime'] and lines[j]['endTime'] == lines[i]['endTime']:
            j += 1
        if j - i > 1 and lines[i]['startTime'] > 0.0:
            seg_start = lines[i]['startTime']
            seg_end   = lines[i]['endTime']
            count     = j - i
            duration  = (seg_end - seg_start) / count
            for k in range(count):
                lines[i + k]['startTime'] = round(seg_start + k * duration, 3)
                lines[i + k]['endTime']   = round(seg_start + (k + 1) * duration, 3)
        i = j


def _flatten_words(word_segments: list[dict]) -> list[tuple[str, float, float]]:
    """Flatten WhisperX word-level data into (char, start, end) tuples."""
    flat: list[tuple[str, float, float]] = []
    for seg in word_segments:
        for w in seg.get('words', []):
            s = w.get('start')
            e = w.get('end')
            c = w.get('word', '')
            if s is not None and e is not None and c:
                flat.append((c, float(s), float(e)))
    return flat


def _norm_chars(text: str) -> str:
    """Normalize text to bare CJK/kana/alnum for character-level matching."""
    s = unicodedata.normalize('NFKC', text)
    s = _kata_to_hira(s)
    s = re.sub(r'[^぀-ゟ一-鿿㐀-䶿a-z0-9]', '', s)
    return s.lower()


# ── Hybrid timing: bridge DTW char-level gaps ──────────────────────────────────

_BRIDGE_THRESHOLD_S = 1.5   # bridge cross-segment gaps smaller than this


def _build_word_timeline(transcript: list[dict]) -> list[tuple[float, float]]:
    """Flat sorted list of (word_start, word_end) from all transcript words."""
    times: list[tuple[float, float]] = []
    for seg in transcript:
        for w in seg.get('words', []):
            s = w.get('start')
            e = w.get('end')
            if s is not None and e is not None:
                times.append((float(s), float(e)))
    times.sort()
    return times


def _has_speech_in_gap(
    word_times: list[tuple[float, float]],
    t_start: float,
    t_end: float,
) -> bool:
    """Return True if any word overlaps the interval [t_start, t_end]."""
    for ws, we in word_times:
        if ws > t_end + 0.05:
            break
        if we >= t_start - 0.05:
            return True
    return False


def _bridge_acoustic_gaps(
    lines:      list[dict],
    transcript: list[dict],
    job_log=None,
) -> list[dict]:
    """
    Eliminate artificial inter-line gaps produced by DTW char-level matching.

    DTW assigns each official lyric line to a WhisperX *segment*, then
    _apply_char_timestamps refines positions using bigram similarity within
    the character stream.  When consecutive lines share the same segment the
    char matcher can leave a gap that has no acoustic basis — the word stream
    is contiguous and no silence exists there.

    Bridging rules:
      1. Same transcriptIndex  → always bridge (gap is a char-match artefact)
      2. Adjacent segment indices, gap < _BRIDGE_THRESHOLD_S, no speech in gap
                               → bridge (small cross-segment transition artefact)
      3. All other cases       → preserve (intentional pause / instrumental)

    The function mutates lines in-place and returns them.
    """
    if len(lines) < 2:
        return lines

    word_times = _build_word_timeline(transcript)

    for i in range(len(lines) - 1):
        curr    = lines[i]
        nxt     = lines[i + 1]
        curr_tx = curr.get('transcriptIndex', -1)
        next_tx = nxt.get('transcriptIndex', -1)
        gap     = round(nxt['startTime'] - curr['endTime'], 3)

        if gap <= 0.05:
            continue   # already contiguous (rounding noise)

        bridged = False
        speech_in_gap = False
        timing_source = nxt.get('timingSource', 'dtw_char')

        if curr_tx >= 0 and curr_tx == next_tx:
            # ── Rule 1: same WhisperX segment ─────────────────────────────────
            nxt['startTime'] = curr['endTime']
            bridged = True
            bridge_reason = f'same_segment(tx={curr_tx})'

        elif (curr_tx >= 0 and next_tx == curr_tx + 1
              and gap < _BRIDGE_THRESHOLD_S):
            # ── Rule 2: adjacent segments, small gap ──────────────────────────
            speech_in_gap = _has_speech_in_gap(
                word_times, curr['endTime'], nxt['startTime']
            )
            if not speech_in_gap:
                nxt['startTime'] = curr['endTime']
                bridged = True
                bridge_reason = (
                    f'adjacent_segs_no_speech'
                    f'(tx={curr_tx}→{next_tx} gap={gap:.3f}s)'
                )
            else:
                bridge_reason = (
                    f'preserved_speech_in_gap'
                    f'(tx={curr_tx}→{next_tx} gap={gap:.3f}s)'
                )
        else:
            bridge_reason = (
                f'preserved_intentional'
                f'(tx={curr_tx}→{next_tx} gap={gap:.3f}s)'
            )

        if job_log:
            job_log.info(
                f"[hybrid] L{i+1:02d}: "
                f"text='{nxt.get('text', '')[:20]}' "
                f"gap_was={gap:.3f}s bridged={bridged} "
                f"speech_in_gap={speech_in_gap} "
                f"reason={bridge_reason} "
                f"new_start={nxt['startTime']:.3f}s",
                stage='align',
            )


def _apply_char_timestamps(
    matched_lines: list[dict],
    word_segments: list[dict],
) -> list[dict]:
    """
    Replace segment-level timestamps with precise character-level timestamps
    by sliding a window over WhisperX's character sequence and matching each
    lyric line's characters to the closest position.

    Falls back to the existing timestamp if no good match is found.
    """
    flat = _flatten_words(word_segments)
    if not flat:
        return matched_lines

    # Build normalised char sequence with per-char timestamps
    tx_chars: list[tuple[str, float, float]] = []
    for word, s, e in flat:
        chars = _norm_chars(word)
        if not chars:
            continue
        # Distribute word timing evenly across its characters
        n = len(chars)
        step = (e - s) / n
        for k, c in enumerate(chars):
            tx_chars.append((c, round(s + k * step, 3), round(s + (k + 1) * step, 3)))

    if not tx_chars:
        return matched_lines

    tx_str    = ''.join(c for c, _, _ in tx_chars)
    tx_cursor = 0
    # Allow scanning a few positions back to recover from tx_cursor overrun.
    # Overrun happens when the previous line's match window extends into the
    # first character of the current line (common with wide-span char tokens).
    _LOOKBACK = 3
    result: list[dict] = []

    for line in matched_lines:
        line_norm = _norm_chars(line.get('text', ''))

        if not line_norm or tx_cursor >= len(tx_chars):
            result.append(line)
            continue

        win       = max(1, len(line_norm))
        look_from = max(0, tx_cursor - _LOOKBACK)
        search_end = min(len(tx_chars), look_from + win + 100)
        search_str = tx_str[look_from:search_end]

        best_pos   = -1
        best_score = 0.0

        for offset in range(max(1, len(search_str) - win + 1)):
            window = search_str[offset:offset + win]
            score  = bigram_sim(line_norm, window)
            if score > best_score:
                best_score = score
                best_pos   = look_from + offset

        if best_pos >= 0 and best_score > 0.12:
            match_end  = min(best_pos + win - 1, len(tx_chars) - 1)
            start_t    = tx_chars[best_pos][1]
            end_t      = tx_chars[match_end][2]
            result.append({**line, 'startTime': start_t, 'endTime': end_t})
            tx_cursor  = max(tx_cursor, match_end + 1)   # never go backward
        else:
            result.append(line)

    return result


def match_lyric_lines(
    transcript:     list[dict],
    official_lines: list[dict],
    dtw_window:     int   = DEFAULT_DTW_WINDOW,
    anchor_thresh:  float = DEFAULT_ANCHOR_THRESH,
    chorus_thresh:  float = DEFAULT_CHORUS_THRESH,
    drift_correct:  bool  = True,
    max_drift:      float = MAX_HARD_DRIFT,
    job_log=None,
) -> dict:
    """
    Align official_lines to transcript timestamps.

    transcript:     [{text, startTime, endTime, words?}]
    official_lines: [{index, text}] or [{text}]
    Returns: dict with 'lines' and 'stats'.
    """
    def log(msg: str) -> None:
        if job_log:
            job_log.info(msg, stage='align')
        logger.info(msg)

    if not transcript or not official_lines:
        return {'lines': [], 'stats': {}, 'anchors': []}

    n = len(official_lines)
    m = len(transcript)

    log(f"Matching {n} official lines to {m} transcript segments")

    # 1. Normalize
    norm_off    = [normalize(l.get('text', '')) for l in official_lines]
    norm_tx     = [normalize(s.get('text', '')) for s in transcript]
    romaji_off  = [to_romaji(s) for s in norm_off]
    romaji_tx   = [to_romaji(s) for s in norm_tx]

    # 2. Chorus detection
    chorus_groups = detect_choruses(norm_off, chorus_thresh)
    chorus_map: dict[int, dict] = {}
    for g in chorus_groups:
        for idx in g['lineIndices']:
            chorus_map[idx] = g
    log(f"Detected {len(chorus_groups)} chorus group(s)")

    # 3. Similarity matrix
    sim_matrix = [
        [compute_sim(norm_off[i], norm_tx[j], romaji_off[i], romaji_tx[j])
         for j in range(m)]
        for i in range(n)
    ]
    cost_matrix = [[1.0 - s for s in row] for row in sim_matrix]

    # 4. DTW
    eff_window = dtw_window if dtw_window > 0 else max(8, int(m * 0.15))
    _, path    = dtw(cost_matrix, eff_window)
    mapping    = resolve_path(path, n, cost_matrix)

    # 5. Raw matches
    raw: list[dict] = []
    for i, off in enumerate(official_lines):
        j = mapping[i]
        if j < 0 or j >= m:
            raw.append({'officialIndex': i, 'transcriptIndex': -1, 'similarity': 0.0,
                        'startTime': 0.0, 'endTime': 0.0, 'isChorus': i in chorus_map})
            continue
        seg = transcript[j]
        raw.append({
            'officialIndex':   i,
            'transcriptIndex': j,
            'similarity':      sim_matrix[i][j],
            'startTime':       seg.get('startTime', seg.get('start_time', seg.get('start', 0.0))),
            'endTime':         seg.get('endTime',   seg.get('end_time',   seg.get('end',   0.0))),
            'isChorus':        i in chorus_map,
        })

    # 6. Drift correction (anchor-based)
    anchors: list[dict] = []
    if drift_correct:
        prev_time = float('-inf')
        for m_item in raw:
            if m_item['transcriptIndex'] < 0: continue
            if m_item['similarity'] >= anchor_thresh and m_item['startTime'] > prev_time:
                anchors.append({
                    'officialIndex':   m_item['officialIndex'],
                    'transcriptIndex': m_item['transcriptIndex'],
                    'similarity':      m_item['similarity'],
                    'expectedTime':    m_item['startTime'],
                    'actualTime':      m_item['startTime'],
                    'correctedOffset': 0.0,
                })
                prev_time = m_item['startTime']

    # 7. Build AlignedLine list
    anchor_set = {a['officialIndex'] for a in anchors}
    lines: list[dict] = []
    for m_item in raw:
        i    = m_item['officialIndex']
        j    = m_item['transcriptIndex']
        seg  = transcript[j] if j >= 0 and j < len(transcript) else None
        off  = official_lines[i]

        # Anchor distance
        anc_dists = [abs(a['officialIndex'] - i) for a in anchors]
        anc_dist  = min(anc_dists) if anc_dists else -1

        # Temporal score
        prev_t = raw[i-1]['startTime'] if i > 0 else 0.0
        next_t = raw[i+1]['startTime'] if i < len(raw)-1 else float('inf')
        temporal = 1.0 if m_item['startTime'] >= prev_t - 0.1 and m_item['startTime'] <= next_t + 1.0 else 0.3

        # Neighbor score
        prev_sim = raw[i-1]['similarity'] if i > 0 else m_item['similarity']
        next_sim = raw[i+1]['similarity'] if i < len(raw)-1 else m_item['similarity']
        neighbor = (prev_sim + m_item['similarity'] + next_sim) / 3

        anchor_bonus = 0.1 if i in anchor_set else 0.0
        confidence   = min(1.0,
            0.50 * m_item['similarity'] +
            0.25 * temporal +
            0.20 * neighbor +
            0.05 * anchor_bonus
        )

        chorus_grp = chorus_map.get(i)
        lines.append({
            'id':              f'line-{i}',
            'text':            off.get('text', ''),
            'transcriptText':  seg.get('text', '') if seg else '',
            'startTime':       round(m_item['startTime'], 3),
            'endTime':         round(m_item['endTime'],   3),
            'words':           seg.get('words', []) if seg else [],
            'confidence':      round(confidence, 3),
            'textSimilarity':  round(m_item['similarity'], 3),
            'temporalScore':   round(temporal, 3),
            'neighborScore':   round(neighbor, 3),
            'transcriptIndex': j,
            'isChorus':        m_item['isChorus'],
            'chorusGroupId':   chorus_grp['id'] if chorus_grp else None,
            'drift':           0.0,
            'anchorDistance':  anc_dist,
        })

    # Refine timestamps using character-level word alignment (more precise than
    # segment-level matching). Falls back to even subdivision if it fails.
    lines = _apply_char_timestamps(lines, transcript)

    # Bridge artificial gaps introduced by char-level bigram matching.
    # Must run BEFORE _subdivide_shared_segments so the subdivision step
    # only sees residual unresolved cases (should be rare after bridging).
    _bridge_acoustic_gaps(lines, transcript, job_log=job_log)

    _subdivide_shared_segments(lines)  # fallback for any remaining shared segments

    avg_conf = sum(l['confidence'] for l in lines) / max(1, len(lines))
    avg_sim  = sum(l['textSimilarity'] for l in lines) / max(1, len(lines))
    flagged  = [i for i, l in enumerate(lines) if l['confidence'] < 0.45]

    stats = {
        'avgConfidence': round(avg_conf, 3),
        'avgSimilarity': round(avg_sim,  3),
        'chorusGroups':  len(chorus_groups),
        'totalAnchors':  len(anchors),
        'flaggedLines':  flagged,
        'unmatched':     [i for i, m in enumerate(raw) if m['transcriptIndex'] < 0],
    }

    log(f"Match complete: avg_conf={avg_conf:.2f} avg_sim={avg_sim:.2f} "
        f"anchors={len(anchors)} flagged={len(flagged)}")

    return {'lines': lines, 'stats': stats, 'anchors': anchors}


# ── File I/O ───────────────────────────────────────────────────────────────────

MATCHED_FILE = 'matched_lyrics.json'


def run_offline_match(
    output_dir:  Path,
    youtube_id:  str,
    job_log=None,
) -> Optional[dict]:
    """
    Load aligned_words.json + lyrics for a video, run match_lyric_lines,
    and save matched_lyrics.json.

    Lyrics source priority:
      1. lyrics_manual.txt  — user-pasted override (plain text, one line per line)
      2. lyrics_cached.json — auto-fetched from providers

    Returns the result dict, or None if inputs are missing.
    """
    aligned_path = output_dir / 'aligned_words.json'
    manual_path  = output_dir / 'lyrics_manual.txt'
    lyrics_path  = output_dir / 'lyrics_cached.json'

    if not aligned_path.exists():
        if job_log:
            job_log.info('Skipping offline match: missing aligned_words.json', stage='align')
        return None

    # Prefer manually pasted lyrics over auto-fetched ones
    if manual_path.exists():
        raw_text = manual_path.read_text(encoding='utf-8').strip()
        official_lines = [
            {'index': i, 'text': ln.strip()}
            for i, ln in enumerate(raw_text.splitlines())
            if ln.strip()
        ]
        source = 'manual'
    elif lyrics_path.exists():
        lyrics_data = json.loads(lyrics_path.read_text(encoding='utf-8'))
        official_lines = [
            {'index': i, 'text': ln.get('text', ln.get('japanese', ''))}
            for i, ln in enumerate(lyrics_data.get('lines', lyrics_data.get('lyrics', [])))
        ]
        source = 'cached'
    else:
        if job_log:
            job_log.info('Skipping offline match: no lyrics available', stage='align')
        return None

    aligned_data = json.loads(aligned_path.read_text(encoding='utf-8'))
    transcript   = aligned_data.get('segments', [])

    if not transcript or not official_lines:
        return None

    if job_log:
        job_log.info(
            f"Matching {len(official_lines)} lines from {source} lyrics "
            f"to {len(transcript)} transcript segments",
            stage='align',
        )

    result = match_lyric_lines(transcript, official_lines, job_log=job_log)
    result['youtubeId']   = youtube_id
    result['matchedAt']   = datetime.now(timezone.utc).isoformat()
    result['lyricsSource'] = source

    out_path = output_dir / MATCHED_FILE
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    if job_log:
        job_log.info(
            f"Wrote matched_lyrics.json ({len(result['lines'])} lines, source={source})",
            stage='align',
        )
    return result
