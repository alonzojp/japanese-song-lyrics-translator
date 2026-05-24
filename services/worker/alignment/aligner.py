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
import unicodedata
from pathlib import Path
from typing import Callable, Optional

from config import WHISPER_LANGUAGE

logger = logging.getLogger(__name__)

AlignedSegments  = list[dict]   # same shape as transcriber output
AlignmentMethod  = str          # whisperx | aeneas | gentle | even

# ── Segment quality thresholds ────────────────────────────────────────────────
_MIN_DUR_PER_LINE   = 0.40   # seconds — below this a segment is "collapsed"
_RMS_HOP_MS         = 50     # ms per RMS window for silence detection
_MIN_SPLIT_SEP      = 1.0    # minimum seconds between detected split points
_WORD_GAP_SPLIT_MS  = 120.0  # inter-word pause threshold for line boundary detection
_MAX_CTC_LAG_S      = 3.0    # VAD anchor skipped when offset ≥ this (legitimate fill)
_MIN_LINE_DISPLAY_S       = 0.25   # lines shorter than this are flagged and corrected
_DRIFT_CORRECT_THRESH     = 0.40   # extend last line when this many seconds remain before seg_end
_MAX_LINE_EXTENSION_MS    = 1200   # hard cap: never extend a line more than this beyond its acoustic end
_INSTRUMENTAL_TAIL_MS     = 1800   # gaps beyond this are classified as instrumental tails
_VOCAL_ENERGY_THRESHOLD   = 0.015  # RMS below this = silence/instrumental (no vocal)

# ── Feature flags ─────────────────────────────────────────────────────────────
# Phase 1: text-informed segment mapping (char bigrams + English anchors + greedy monotone).
# Set False to revert to duration-only assignment.
USE_TEXT_AWARE_SEGMENT_MAPPING = True

# Minimum score advantage for the text mapper to advance to the next segment.
# Prevents spurious advances when all segments look similar (poor transcript).
_TEXT_ADVANCE_BIAS = 0.15

# When composite similarity falls below this threshold the mapper treats the
# line as ambiguous and logs a LOW_CONFIDENCE warning.
_TEXT_LOW_CONF_THRESHOLD = 0.12


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


# ── Official lyrics forced alignment ──────────────────────────────────────────

def _char_count(text: str) -> int:
    """Count non-whitespace characters after NFKC normalization."""
    return len(re.sub(r'\s', '', unicodedata.normalize('NFKC', text)))


def _split_words_into_lines(
    words:      list[dict],
    line_texts: list[str],
    parent_seg: dict,
) -> list[dict]:
    """
    Distribute word-level timestamps proportionally across lyric lines
    based on character count. Falls back to even time split if no words.
    """
    if not words:
        # Character-proportional distribution — longer lines get more time.
        # Previously used equal slices (dur/n), which ignored character count.
        return _proportional_line(parent_seg, line_texts)

    counts  = [max(1, _char_count(t)) for t in line_texts]
    total_c = sum(counts)
    total_w = len(words)
    result  = []
    widx    = 0

    for i, (text, count) in enumerate(zip(line_texts, counts)):
        if i == len(line_texts) - 1:
            line_words = words[widx:]
        else:
            n = max(1, round(count / total_c * total_w))
            line_words = words[widx:widx + n]
            widx += n

        if line_words:
            # For the first line in a segment, use the Whisper segment boundary rather
            # than the first word's CTC timestamp. The VAD boundary is closer to actual
            # vocal onset; CTC timestamps lag ~0.5-2s behind the audible voice.
            start_t = (
                parent_seg.get('start', 0.0)
                if i == 0
                else line_words[0].get('start', parent_seg.get('start', 0.0))
            )
            result.append({
                'text':  text,
                'start': round(start_t, 3),
                'end':   round(line_words[-1].get('end', parent_seg.get('end', 0.0)), 3),
                'words': line_words,
            })
        else:
            result.append({
                'text':  text,
                'start': parent_seg.get('start', 0.0),
                'end':   parent_seg.get('end',   0.0),
                'words': [],
            })

    return result


# ── Text-informed segment mapping (Phase 1 + 3) ──────────────────────────────

# Common J-pop English loanwords → their expected hiragana form in ASR transcripts.
# Used to match lyrics like "Tune" against transcript katakana like "チューン".
# Entries are normalized (katakana→hiragana, long vowel ー kept as-is).
_KANA_LOANWORDS: dict[str, str] = {
    # Wonder Scale specific
    "tune":     "ちゅーん",
    "wonder":   "わんだー",
    "scale":    "すけーる",
    "stage":    "すてーじ",
    # Common music/performance
    "soul":     "そうる",
    "song":     "そんぐ",
    "dance":    "だんす",
    "music":    "みゅーじっく",
    "heart":    "はーと",
    "star":     "すたー",
    "shine":    "しゃいん",
    "smile":    "すまいる",
    "voice":    "ぼいす",
    "dream":    "どりーむ",
    "love":     "らぶ",
    "world":    "わーるど",
    "light":    "らいと",
    "night":    "ないと",
    "time":     "たいむ",
    "life":     "らいふ",
    "free":     "ふりー",
    "true":     "とぅるー",
    "false":    "ふぁるす",
    "happy":    "はっぴー",
    "baby":     "べいびー",
    "magic":    "まじっく",
    "miracle":  "みらくる",
    "memory":   "めもりー",
    "believe":  "びりーぶ",
    "fantasy":  "ふぁんたじー",
    "lonely":   "ろーんりー",
    "tender":   "てんだー",
    "color":    "からー",
    "colour":   "からー",
    "chance":   "ちゃんす",
    "power":    "ぱわー",
    "flower":   "ふらわー",
    "story":    "すとーりー",
    "feeling":  "ふぃーりんぐ",
    "morning":  "もーにんぐ",
    "moon":     "むーん",
    "deep":     "でぃーぷ",
    "super":    "すーぱー",
    "cool":     "くーる",
    "special":  "すぺしゃる",
    "perfect":  "ぱーふぇくと",
    "lucky":    "らっきー",
    "crazy":    "くれいじー",
    "hero":     "ひーろー",
    "zero":     "ぜろ",
    "summer":   "さまー",
    "winter":   "うぃんたー",
    "spring":   "すぷりんぐ",
    "speed":    "すぴーど",
    "limit":    "りみっと",
    "rainbow":  "れいんぼー",
    "natural":  "なちゅらる",
    "forever":  "ふぉーえばー",
    "together": "とぅぎゃざー",
    "missing":  "みっしんぐ",
    "shining":  "しゃいにんぐ",
    "running":  "らんにんぐ",
    "angel":    "えんじぇる",
    "fire":     "ふぁいあ",
    "rain":     "れいん",
    "sky":      "すかい",
    "girl":     "がーる",
    "boy":      "ぼーい",
    "wonder":   "わんだー",
    "state":    "すてーと",
    "wonder":   "わんだー",
    # Common single-syllable words
    "my":       "まい",
    "your":     "ゆあ",
    "new":      "にゅー",
    "now":      "なう",
    "go":       "ごー",
    "oh":       "おー",
    "hey":      "へい",
    "yeah":     "いぇあ",
}


def _normalize_for_matching(text: str) -> str:
    """
    Normalize text for cross-script similarity comparison.
    - Katakana → hiragana (U+30A1–U+30F6, offset 0x60)
    - NFKC: full-width → half-width, composed forms
    - Lowercase
    - Strip punctuation; keep CJK, kana, alphanumeric
    """
    chars = []
    for ch in text:
        cp = ord(ch)
        if 0x30A1 <= cp <= 0x30F6:      # katakana ァ–ヶ → hiragana
            chars.append(chr(cp - 0x60))
        else:
            chars.append(ch)
    text = unicodedata.normalize('NFKC', ''.join(chars)).lower()
    # Keep: CJK (4E00-9FFF), hiragana (3040-309F), katakana remnants (30A0-30FF),
    #       ASCII alphanumeric.  Drop everything else (punctuation, whitespace, etc.)
    text = re.sub(r'[^぀-ヿ一-鿿＀-￯a-z0-9]', '', text)
    return text


def _char_bigrams(text: str) -> frozenset[str]:
    """Return the set of consecutive character bigrams from normalized text."""
    n = _normalize_for_matching(text)
    if len(n) < 2:
        return frozenset()
    return frozenset(n[i:i + 2] for i in range(len(n) - 1))


def _english_tokens(text: str) -> frozenset[str]:
    """Return lower-cased English word tokens (letter sequences only)."""
    return frozenset(w.lower() for w in re.findall(r'[A-Za-z]+', text))


def _kana_loanword_sim(lyric: str, segment_text: str) -> float:
    """
    Phase 3 — katakana-English equivalence signal.

    Handles the common J-pop case where the official lyric uses English
    ("Tune", "Wonder Scale") but the ASR transcript uses katakana
    ("チューン", "ワンダーステージ").

    For each English token in the lyric, looks up its expected hiragana form
    in _KANA_LOANWORDS and checks whether that form appears as a substring of
    the normalized segment text.  Returns the fraction of English tokens that
    matched via the kana route (0.0 if the lyric has no English).

    This is a recall-oriented score: it rewards segments that contain the
    katakana reading of the lyric's English words, even when the segment
    contains no Latin characters at all.
    """
    lyric_eng = _english_tokens(lyric)
    if not lyric_eng:
        return 0.0

    seg_norm = _normalize_for_matching(segment_text)
    if not seg_norm:
        return 0.0

    matches = sum(
        1 for eng in lyric_eng
        if (kana := _KANA_LOANWORDS.get(eng)) and kana in seg_norm
    )
    return matches / len(lyric_eng)


def _text_similarity(lyric: str, segment_text: str) -> float:
    """
    Composite similarity score in [0, 1] for one (lyric line, transcript segment) pair.

    Three signals combined:
    - char_sim:     bigram recall over normalized text (kana + kanji + latin)
    - eng_sim:      English token recall — lyric English words found verbatim in segment
    - kana_sim:     katakana-English equivalence — lyric English words found as their
                    expected katakana readings in the segment (Phase 3)

    eng_combined = max(eng_sim, kana_sim) ensures that whether the transcript
    writes a word as English or as katakana, the match is credited.

    When the lyric contains English, English/kana combined dominates (0.40/0.60).
    Pure-Japanese lyrics use char_sim alone.
    """
    lyric_bg   = _char_bigrams(lyric)
    seg_bg     = _char_bigrams(segment_text)
    char_sim   = len(lyric_bg & seg_bg) / len(lyric_bg) if lyric_bg else 0.0

    lyric_eng  = _english_tokens(lyric)
    seg_eng    = _english_tokens(segment_text)
    eng_sim    = len(lyric_eng & seg_eng) / len(lyric_eng) if lyric_eng else 0.0

    if lyric_eng:
        kana_sim     = _kana_loanword_sim(lyric, segment_text)
        eng_combined = max(eng_sim, kana_sim)
        score = 0.40 * char_sim + 0.60 * eng_combined
    else:
        score = char_sim

    return round(score, 4)


def _map_lines_to_segments_by_text(
    rough_segments: AlignedSegments,
    official_lines: list[dict],
    job_log,
) -> dict[int, list[str]]:
    """
    Assign official lyric lines to transcript segments using text similarity.

    Phase 2 — globally optimal monotone assignment via dynamic programming.

    dp[i][j] = best total score assigning lines 0..i-1 to segments 0..j-1.

    Transition (for each cut point k):
        group_score = mean(sim[t][j-1] for t in range(k, i))   # MEAN not sum
        dp[i][j]   = max(dp[k][j-1] + group_score)

    Using mean rather than sum prevents the DP from stuffing many lines into
    one high-scoring segment.  Ordering is enforced structurally — the DP
    never assigns a later line to an earlier segment.

    Complexity: O(N² × M) — trivial at song scale (~23 lines × 10 segments).

    Falls back to duration mapping when no feasible assignment exists.
    """
    N = len(official_lines)
    M = len(rough_segments)

    if N == 0 or M == 0:
        return {}

    seg_texts = [s.get('text', '') for s in rough_segments]

    # ── Build full similarity matrix ──────────────────────────────────────────
    sim: list[list[float]] = [
        [_text_similarity(official_lines[i]['text'], seg_texts[j]) for j in range(M)]
        for i in range(N)
    ]

    # ── Log full matrix at DEBUG level ────────────────────────────────────────
    if job_log:
        job_log.info(
            f"[assign] Text similarity matrix ({N} lines × {M} segs) — "
            f"top-2 scores per line:",
            stage="transcribe",
        )
        for i, line in enumerate(official_lines):
            ranked = sorted(enumerate(sim[i]), key=lambda x: -x[1])
            top2   = " | ".join(f"S{j}={s:.2f}" for j, s in ranked[:2])
            job_log.info(
                f"[assign]   L{i:02d} '{line['text'][:28]}': {top2}",
                stage="transcribe",
            )

    # ── DP: globally optimal monotone assignment ─────────────────────────────
    # dp[i][j] = best total score assigning lines 0..i-1 to segments 0..j-1
    # bp[i][j] = cut point k* that achieved dp[i][j]
    _NEG_INF = float('-inf')

    dp = [[_NEG_INF] * (M + 1) for _ in range(N + 1)]
    bp = [[0]         * (M + 1) for _ in range(N + 1)]
    dp[0][0] = 0.0

    for j in range(1, M + 1):         # consider segments 0..j-1
        for i in range(0, N + 1):     # first i lines assigned
            # Segment j-1 receives lines k..i-1 (k==i → segment is empty)
            for k in range(0, i + 1):
                prev = dp[k][j - 1]
                if prev == _NEG_INF:
                    continue
                if k == i:
                    group_score = 0.0   # empty segment
                else:
                    group_score = (
                        sum(sim[t][j - 1] for t in range(k, i)) / (i - k)
                    )
                candidate = prev + group_score
                if candidate > dp[i][j]:
                    dp[i][j] = candidate
                    bp[i][j] = k

    optimal_score = dp[N][M]

    # ── Reconstruct assignment from backpointers ──────────────────────────────
    groups: list[tuple[int, int, int]] = []   # (seg_idx, line_start, line_end)
    i_cur, j_cur = N, M
    while j_cur > 0:
        k_star = bp[i_cur][j_cur]
        groups.append((j_cur - 1, k_star, i_cur))
        i_cur  = k_star
        j_cur -= 1
    groups.reverse()

    # Build output and collect per-line stats
    seg_to_lines:   dict[int, list[str]] = {}
    low_conf_lines: list[int]            = []

    for seg_idx, start, end in groups:
        if start == end:
            continue   # segment received no lines
        texts: list[str] = []
        for t in range(start, end):
            score = sim[t][seg_idx]
            texts.append(official_lines[t]['text'])
            if score < _TEXT_LOW_CONF_THRESHOLD:
                low_conf_lines.append(t)
        seg_to_lines[seg_idx] = texts

    # ── Log DP result ─────────────────────────────────────────────────────────
    if job_log:
        job_log.info(
            f"[assign] DP optimal score={optimal_score:.4f} "
            f"({N} lines, {M} segs)",
            stage="transcribe",
        )
        for seg_idx, start, end in groups:
            if start == end:
                continue
            n_grp    = end - start
            mean_sim = sum(sim[t][seg_idx] for t in range(start, end)) / n_grp
            job_log.info(
                f"[assign]   Seg{seg_idx}: lines {start}–{end - 1} "
                f"({n_grp} lines, mean_sim={mean_sim:.2f})",
                stage="transcribe",
            )
            for t in range(start, end):
                score = sim[t][seg_idx]
                flag  = " ⚑LOW_CONF" if score < _TEXT_LOW_CONF_THRESHOLD else ""
                job_log.info(
                    f"[assign]     L{t:02d} sim={score:.2f}{flag}  "
                    f"'{official_lines[t]['text'][:32]}'",
                    stage="transcribe",
                )

    if low_conf_lines and job_log:
        job_log.warning(
            f"[assign] LOW_CONFIDENCE on {len(low_conf_lines)} line(s): "
            f"{low_conf_lines} — transcript may be too garbled for text mapping",
            stage="transcribe",
        )

    if job_log:
        lines_per_seg = {seg_idx: end - start
                         for seg_idx, start, end in groups if start < end}
        job_log.info(
            f"[assign] DP assignment complete (score={optimal_score:.3f}): "
            f"{lines_per_seg}",
            stage="transcribe",
        )

    return seg_to_lines


def _map_lines_to_segments_by_duration(
    rough_segments: AlignedSegments,
    official_lines: list[dict],
) -> dict[int, list[str]]:
    """
    Map lyric lines to Whisper segments proportionally by segment duration.

    More reliable than DTW when transcription text quality is low (~35%
    similarity) because it uses audio timing rather than text matching.
    A longer segment gets more lyric lines; gaps (instrumentals) are
    automatically respected because line fractions skip over them.
    """
    n = len(official_lines)
    m = len(rough_segments)

    durations  = [max(0.1, s.get('end', 0) - s.get('start', 0)) for s in rough_segments]
    total_dur  = sum(durations)

    # Cumulative fraction at the START of each segment
    cum: list[float] = [0.0]
    for d in durations:
        cum.append(cum[-1] + d / total_dur)

    seg_to_lines: dict[int, list[str]] = {}
    for i, line in enumerate(official_lines):
        frac = (i + 0.5) / n          # position of this line in the song
        # find segment whose range contains frac
        assigned = m - 1
        for j in range(m):
            if cum[j] <= frac < cum[j + 1]:
                assigned = j
                break
        seg_to_lines.setdefault(assigned, []).append(line['text'])

    return seg_to_lines


def _proportional_line(
    seg:        dict,
    line_texts: list[str],
) -> list[dict]:
    """
    Distribute a segment's time range across lyric lines proportionally
    by character count. Reliable fallback when acoustic alignment drifts.
    """
    start  = seg.get('start', 0.0)
    end    = seg.get('end',   0.0)
    counts = [max(1, _char_count(t)) for t in line_texts]
    total  = sum(counts)
    result = []
    t      = start
    for text, count in zip(line_texts, counts):
        dur = (end - start) * count / total
        result.append({
            'text':  text,
            'start': round(t, 3),
            'end':   round(t + dur, 3),
            'words': [],
        })
        t += dur
    return result


def _enforce_min_durations(
    lines:   list[dict],
    min_dur: float,
    seg_end: float,
    job_log=None,
    seg_id:  int = -1,
) -> list[dict]:
    """
    Enforce a minimum display duration on every lyric line.

    Scans forward; when a line is too short it steals time from the next line
    (advances next_start) up to the point where the next line would itself fall
    below min_dur.  Any line that cannot be fixed is logged as a warning.

    Must run AFTER word-gap splitting and BEFORE gap bridging so the corrected
    boundaries feed into the final timeline.
    """
    if len(lines) <= 1:
        return lines

    result = [dict(ln) for ln in lines]

    for i in range(len(result) - 1):   # last line handled by drift correction
        dur = result[i].get('end', 0.0) - result[i].get('start', 0.0)
        if dur >= min_dur:
            continue

        next_end   = result[i + 1].get('end',   0.0)
        next_start = result[i + 1].get('start', 0.0)
        next_dur   = next_end - next_start
        can_give   = max(0.0, next_dur - min_dur)
        need       = min_dur - dur
        give       = min(need, can_give)

        if give > 0.005:
            new_end        = round(result[i]['end']       + give, 3)
            new_next_start = round(result[i + 1]['start'] + give, 3)
            if job_log:
                job_log.info(
                    f"[min_dur] Seg{seg_id} line{i}: dur={dur:.3f}s < {min_dur}s "
                    f"— end {result[i]['end']:.3f}→{new_end:.3f}s  "
                    f"next_start {result[i+1]['start']:.3f}→{new_next_start:.3f}s",
                    stage="transcribe",
                )
            result[i]['end']       = new_end
            result[i + 1]['start'] = new_next_start
            dur = new_end - result[i]['start']

        if dur < min_dur and job_log:
            job_log.warning(
                f"[min_dur] Seg{seg_id} line{i}: dur={dur:.3f}s still < {min_dur}s "
                f"— '{result[i].get('text', '')[:32]}'",
                stage="transcribe",
            )

    return result


def _split_by_word_gaps(
    words:      list[dict],
    line_texts: list[str],
    parent_seg: dict,
    min_gap_ms: float = _WORD_GAP_SPLIT_MS,
    job_log     = None,
    seg_id:     int    = -1,
) -> list[dict]:
    """
    Split word tokens into lyric lines using acoustic word boundary timing.

    Priority order:
      1. Inter-word pause detection — ALWAYS uses the n-1 largest gaps as split
         points, with NO minimum threshold.  Even a 10ms pause is more reliable
         than arbitrary word-count arithmetic.  Gaps are classified as 'strong'
         (≥ min_gap_ms) or 'weak' (best available) in the debug log.
      2. Character-weighted word allocation — fallback when the word sequence has
         fewer gaps than needed (e.g. very dense singing or word-count < n).
         Allocates words proportionally to lyric character counts so that longer
         lines receive proportionally more time.  Always better than even count.
      3. Proportional line timing — only when no word timestamps exist at all.

    Removing the minimum-gap threshold eliminates cumulative timing drift:
    the largest gap in a phrase group almost always corresponds to the real
    lyric line boundary, even when singers don't pause explicitly.
    """
    n = len(line_texts)

    if not words:
        if job_log:
            job_log.info(
                f"[split] Seg{seg_id}: no word timestamps — proportional fallback",
                stage="transcribe",
            )
        return _proportional_line(parent_seg, line_texts)

    if n == 1:
        return [{
            'text':  line_texts[0],
            'start': round(float(words[0].get('start') or parent_seg.get('start', 0.0)), 3),
            'end':   round(float(words[-1].get('end')   or parent_seg.get('end',   0.0)), 3),
            'words': words,
        }]

    # ── Compute all inter-word gaps ────────────────────────────────────────────
    gaps: list[tuple[float, int]] = []   # (gap_ms, word_index_before_gap)
    for idx in range(len(words) - 1):
        w_end   = words[idx].get('end')
        w_start = words[idx + 1].get('start')
        if w_end is not None and w_start is not None:
            gaps.append(((float(w_start) - float(w_end)) * 1000, idx))

    # ── Split decision reasoning ───────────────────────────────────────────────
    if job_log and n > 1:
        if gaps:
            gap_vals = [g for g, _ in gaps]
            job_log.info(
                f"[split_reason] Seg{seg_id}: {n} lines, {len(words)} tokens  "
                f"gaps={len(gap_vals)}  mean={sum(gap_vals)/len(gap_vals):.0f}ms  "
                f"max={max(gap_vals):.0f}ms  min={min(gap_vals):.0f}ms  "
                f"need={n-1} split(s)",
                stage="transcribe",
            )
        else:
            job_log.info(
                f"[split_reason] Seg{seg_id}: {n} lines, {len(words)} tokens  "
                f"no gaps available → char_weighted fallback",
                stage="transcribe",
            )

    # ── Select n-1 split points ────────────────────────────────────────────────
    if len(gaps) >= n - 1:
        # Use the n-1 largest gaps — no minimum threshold.
        # Any acoustic gap outperforms word-count arithmetic.
        top        = sorted(gaps, key=lambda x: -x[0])[:n - 1]
        split_at   = sorted(idx for _, idx in top)
        chosen_gap_ms = [g for g, i in sorted(top, key=lambda x: split_at.index(x[1]))]
        method     = (
            'gap_strong' if all(g >= min_gap_ms for g in chosen_gap_ms)
            else 'gap_weak'
        )
        if job_log and n > 1:
            job_log.info(
                f"[split_reason] Seg{seg_id}: chose {method}  "
                f"selected_gaps={[round(g,0) for g in chosen_gap_ms]}ms",
                stage="transcribe",
            )
    else:
        # Fallback: character-weighted allocation.
        # Longer lyric lines are allocated proportionally more word tokens.
        counts    = [max(1, _char_count(t)) for t in line_texts]
        total_c   = sum(counts)
        total_w   = len(words)
        split_at  = []
        word_pos  = 0
        for cnt in counts[:-1]:
            word_pos += max(1, round(cnt / total_c * total_w))
            split_at.append(min(max(0, word_pos - 1), total_w - 2))
        chosen_gap_ms = []
        method = 'char_weighted'
        if job_log and n > 1:
            job_log.info(
                f"[split_reason] Seg{seg_id}: fell back to char_weighted  "
                f"(only {len(gaps)} gaps, need {n-1})",
                stage="transcribe",
            )

    # ── Build lines from word groups ───────────────────────────────────────────
    edges  = [0] + [s + 1 for s in split_at] + [len(words)]
    result: list[dict] = []

    for i, text in enumerate(line_texts):
        grp = words[edges[i]:edges[i + 1]]
        if grp:
            start_t = float(grp[0].get('start') or parent_seg.get('start', 0.0))
            end_t   = float(grp[-1].get('end')   or parent_seg.get('end',   0.0))
        else:
            # More lines than word tokens — proportional emergency fallback
            seg_dur = max(0.001, parent_seg.get('end', 0.0) - parent_seg.get('start', 0.0))
            start_t = parent_seg.get('start', 0.0) + i * seg_dur / n
            end_t   = start_t + seg_dur / n

        # Gap between this line's last word and the next line's first word
        boundary_gap_ms: Optional[float] = None
        if i < n - 1 and edges[i + 1] < len(words) and grp:
            nw_start = words[edges[i + 1]].get('start')
            lw_end   = grp[-1].get('end')
            if nw_start is not None and lw_end is not None:
                boundary_gap_ms = round((float(nw_start) - float(lw_end)) * 1000, 1)

        line_dur = round(end_t - start_t, 3)
        dur_flag = ' ⚑SHORT' if line_dur < _MIN_LINE_DISPLAY_S else ''

        # Boundary confidence for the transition OUT of this line to the next
        b_conf:  Optional[float] = None
        b_label: str             = ''
        if i < n - 1:
            b_conf, b_label = _boundary_conf_score(boundary_gap_ms, method)

        line_dict: dict = {
            'text':  text,
            'start': round(start_t, 3),
            'end':   round(end_t,   3),
            'words': grp,
        }
        if b_conf is not None:
            line_dict['_boundary_conf']       = b_conf
            line_dict['_boundary_conf_label'] = b_label

        result.append(line_dict)

        if job_log:
            gap_str  = f" next_gap={boundary_gap_ms:.0f}ms" if boundary_gap_ms is not None else ""
            conf_str = f" bconf={b_conf:.2f}({b_label})" if b_conf is not None else ""
            job_log.info(
                f"[split] Seg{seg_id} line{i}: {start_t:.3f}–{end_t:.3f}s "
                f"dur={line_dur:.3f}s words={len(grp)} method={method}"
                f"{gap_str}{conf_str}{dur_flag}  '{text[:28]}'",
                stage="transcribe",
            )

    if job_log and n > 1:
        split_times = [
            round(words[s].get('end', 0), 3)
            for s in split_at if s < len(words)
        ]
        job_log.info(
            f"[split] Seg{seg_id}: {n} lines, method={method}, "
            f"split_after_word_ends={split_times}",
            stage="transcribe",
        )

    # Enforce minimum line duration — steal time from adjacent lines when a
    # short gap produces a line too brief to read (<_MIN_LINE_DISPLAY_S).
    result = _enforce_min_durations(
        result, _MIN_LINE_DISPLAY_S, parent_seg.get('end', 0.0), job_log, seg_id
    )

    # Token-to-line assignment map and acoustic vs proportional comparison
    _log_token_map(result, seg_id, job_log)
    if n > 1 and job_log:
        _compare_splits(result, _proportional_line(parent_seg, line_texts), seg_id, job_log)

    return result


def _apply_vad_anchor(
    words:     list[dict],
    vad_start: float,
    seg_id:    int,
    job_log,
) -> tuple[list[dict], float]:
    """
    Shift all word timestamps backward so the first word starts at vad_start.

    WhisperX CTC alignment places the first word 100-2500ms after the actual
    acoustic onset (VAD start).  This offset is pure model latency, not a real
    gap — removing it re-anchors all lines in the segment to the true onset.

    Skipped when offset >= _MAX_CTC_LAG_S: that indicates a legitimate
    instrumental fill before singing starts, not CTC lag.

    Returns (corrected_words, applied_offset_ms).  offset_ms == 0 means skipped.
    """
    if not words:
        return words, 0.0
    first_t = words[0].get('start')
    if first_t is None:
        return words, 0.0

    offset = float(first_t) - vad_start
    if offset <= 0:
        return words, 0.0
    if offset >= _MAX_CTC_LAG_S:
        if job_log:
            job_log.info(
                f"[anchor] Seg {seg_id}: skip VAD anchor — "
                f"offset={offset*1000:.0f}ms ≥ {_MAX_CTC_LAG_S*1000:.0f}ms "
                f"(treating as instrumental fill)",
                stage="transcribe",
            )
        return words, 0.0

    corrected = []
    for w in words:
        cw = dict(w)
        if cw.get('start') is not None:
            cw['start'] = round(float(cw['start']) - offset, 3)
        if cw.get('end') is not None:
            cw['end'] = round(float(cw['end']) - offset, 3)
        corrected.append(cw)

    if job_log:
        job_log.info(
            f"[anchor] Seg {seg_id}: VAD anchor applied "
            f"offset={offset*1000:.0f}ms "
            f"(first_word {float(first_t):.3f}s → {vad_start:.3f}s)",
            stage="transcribe",
        )
    return corrected, round(offset * 1000, 1)


def _find_silence_splits(
    audio_path: Path,
    start:      float,
    end:        float,
    n_splits:   int,
) -> list[float]:
    """
    Find n_splits split timestamps within [start, end] at the audio's quietest moments.

    Loads the audio window, computes RMS energy in _RMS_HOP_MS windows, then picks the
    n_splits quietest non-overlapping positions separated by at least _MIN_SPLIT_SEP.

    Returns a sorted list of timestamps, or [] if the segment is too short or
    not enough distinct silences are found.

    Works on the vocals_asr.wav path (16 kHz mono) where Demucs has already
    removed the backing track — vocal pauses are more prominent there.
    """
    if n_splits <= 0:
        return []
    if end - start < (n_splits + 1) * _MIN_SPLIT_SEP:
        return []

    try:
        import soundfile as sf
        import numpy as np

        info     = sf.info(str(audio_path))
        sr       = info.samplerate
        s_frame  = int(start * sr)
        n_frames = int((end - start) * sr)

        audio, sr = sf.read(str(audio_path), start=s_frame, frames=n_frames, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        hop   = max(1, int(_RMS_HOP_MS * sr / 1000))
        win   = hop * 2
        n_win = max(1, (len(audio) - win) // hop)
        if n_win < n_splits + 2:
            return []

        rms = np.array([
            float(np.sqrt(np.mean(audio[i * hop : i * hop + win] ** 2) + 1e-9))
            for i in range(n_win)
        ])

        # Exclude first and last 10% to avoid segment-edge artifacts
        margin    = max(1, n_win // 10)
        min_sep_w = max(1, int(_MIN_SPLIT_SEP * sr / hop))

        # Greedily pick n_splits quietest windows with minimum separation
        sorted_idx = list(np.argsort(rms[margin : n_win - margin]) + margin)
        chosen: list[int] = []
        for idx in sorted_idx:
            if len(chosen) >= n_splits:
                break
            if all(abs(idx - c) >= min_sep_w for c in chosen):
                chosen.append(idx)

        if len(chosen) < n_splits:
            return []

        def to_time(i: int) -> float:
            return round(start + (i * hop + win // 2) / sr, 3)

        return sorted(to_time(i) for i in chosen)

    except Exception:
        return []


def _silence_split_fallback(
    seg:        dict,
    line_texts: list[str],
    audio_path: Path,
    job_log=None,
) -> list[dict]:
    """
    Distribute lyric lines within a segment using silence detection.

    Preferred over _proportional_line() when acoustic alignment fails: finds actual
    quiet moments in the vocal audio rather than guessing by character count.
    Falls back to _proportional_line() if silence detection finds no clean splits.
    """
    n = len(line_texts)
    if n <= 1:
        return _proportional_line(seg, line_texts)

    start  = seg.get('start', 0.0)
    end    = seg.get('end',   0.0)
    splits = _find_silence_splits(audio_path, start, end, n - 1)

    if len(splits) == n - 1:
        boundaries = [start] + splits + [end]
        if job_log:
            job_log.info(
                f"[align] Silence split: {n} lines in {end - start:.1f}s at {start:.1f}s"
                f" — splits at {[round(s, 1) for s in splits]}",
                stage="transcribe",
            )
        return [
            {
                'text':  text,
                'start': round(boundaries[i],     3),
                'end':   round(boundaries[i + 1], 3),
                'words': [],
            }
            for i, text in enumerate(line_texts)
        ]

    # Silence detection failed — fall back to character-count proportional
    return _proportional_line(seg, line_texts)


def _boundary_from_chars(
    chars:      list[dict],
    line_texts: list[str],
    seg_start:  float,
    seg_end:    float,
) -> list[dict]:
    """
    Use WhisperX character-level alignment to find precise lyric line starts.

    The segment text was ' '.join(line_texts). WhisperX may include or exclude
    the joining spaces in its chars output, so we detect which and adjust:
      - With spaces:    len(chars) >= len(' '.join(line_texts))
      - Without spaces: len(chars) >= len(''.join(line_texts))

    Returns a list of line dicts (same length as line_texts), or [] to signal
    that char data is incomplete and the caller should fall back.
    """
    if not chars or not line_texts:
        return []

    full_with_spaces = ' '.join(line_texts)
    full_no_spaces   = ''.join(line_texts)

    if len(chars) >= len(full_with_spaces):
        # Spaces included — build positions with space offsets
        positions: list[int] = []
        pos = 0
        for i, lt in enumerate(line_texts):
            if i > 0:
                pos += 1   # joining space
            positions.append(pos)
            pos += len(lt)
    elif len(chars) >= len(full_no_spaces):
        # Spaces NOT included — positions without space offsets
        positions = [sum(len(lt) for lt in line_texts[:i]) for i in range(len(line_texts))]
    else:
        return []   # insufficient char data

    result: list[dict] = []
    tolerance = 2.0

    for i, (lpos, text) in enumerate(zip(positions, line_texts)):
        if lpos >= len(chars):
            return []
        c          = chars[lpos]
        line_start = c.get('start')
        if line_start is None:
            return []

        # End = start of next line's first char, or last char of this line
        if i + 1 < len(positions):
            next_lpos = positions[i + 1]
            nc        = chars[next_lpos] if next_lpos < len(chars) else None
            line_end  = float(nc['start']) if nc and nc.get('start') is not None else seg_end
        else:
            last_lpos = min(lpos + len(text) - 1, len(chars) - 1)
            line_end  = float(chars[last_lpos].get('end') or seg_end)

        # Reject if outside the segment window
        if not (seg_start - tolerance <= float(line_start) <= seg_end + tolerance):
            return []

        result.append({
            'text':  text,
            'start': round(float(line_start), 3),
            'end':   round(float(line_end),   3),
            'words': [],
        })

    return result


# ── Reconciliation constants ───────────────────────────────────────────────────
_RECONCILE_MIN_OVERLAP = 0.30   # aligned_seg must have ≥30% overlap with rough_seg


def _reconcile_by_overlap(
    rough_segments: AlignedSegments,
    aligned_segs:   AlignedSegments,
) -> tuple[dict[int, list[dict]], list[dict]]:
    """
    Map each rough segment to the aligned segments that fall within its time
    window, matching by time overlap rather than array index.

    When WhisperX resegments internally (e.g. splits one 29-second window into
    three sub-segments) this recovers the char/word data from each sub-segment
    and assigns it to the correct rough segment rather than discarding it.

    Returns ({rough_idx: [aligned_seg, ...]}, unmatched_segs).
    Each bucket is sorted by start time.
    """
    mapping: dict[int, list[dict]] = {i: [] for i in range(len(rough_segments))}
    unmatched: list[dict] = []

    for aseg in aligned_segs:
        a_start = aseg.get('start', 0.0)
        a_end   = aseg.get('end',   0.0)
        a_dur   = max(0.001, a_end - a_start)

        best_i    = -1
        best_frac = -1.0

        for i, rseg in enumerate(rough_segments):
            r_start = rseg.get('start', 0.0)
            r_end   = rseg.get('end',   0.0)
            overlap = max(0.0, min(a_end, r_end) - max(a_start, r_start))
            frac    = overlap / a_dur
            if frac > best_frac:
                best_frac = frac
                best_i    = i

        if best_i >= 0:
            mapping[best_i].append(aseg)
        else:
            unmatched.append(aseg)

    # Sort each bucket by start time
    for i in mapping:
        mapping[i].sort(key=lambda s: s.get('start', 0.0))

    return mapping, unmatched


def _collect_chars(seg_list: list[dict]) -> list[dict]:
    """Merge and sort chars from multiple aligned sub-segments."""
    chars: list[dict] = []
    for seg in seg_list:
        chars.extend(seg.get('chars', []))
    chars.sort(key=lambda c: c.get('start') or 0.0)
    return chars


def _collect_words(seg_list: list[dict]) -> list[dict]:
    """Merge and sort words from multiple aligned sub-segments."""
    words: list[dict] = []
    for seg in seg_list:
        words.extend(seg.get('words', []))
    words.sort(key=lambda w: w.get('start') or 0.0)
    return words


def _align_whisperx_with_official_lyrics(
    rough_segments: AlignedSegments,
    official_lines: list[dict],
    audio_path:     Path,
    job_log,
) -> tuple[AlignedSegments, AlignmentMethod]:
    """
    Assign precise timestamps to official lyric lines using Whisper's timing
    windows.

    Strategy:
      1. Map lines to segments by duration (proportional, no text matching)
      2. WhisperX acoustic alignment with char-level timestamps
      3. Reconcile aligned output → rough segments by time overlap (not index)
      4. Per rough segment: char → word → silence → proportional priority
      5. Validate, bridge gaps, log timing source breakdown
    """
    import whisperx

    device = _get_device()

    if job_log:
        job_log.info(
            f"[align] Official lyrics alignment: "
            f"{len(official_lines)} lines → {len(rough_segments)} windows",
            stage="transcribe",
        )

    # ── 1. Segment-to-lyric assignment ───────────────────────────────────────
    if USE_TEXT_AWARE_SEGMENT_MAPPING:
        seg_to_lines = _map_lines_to_segments_by_text(rough_segments, official_lines, job_log)
    else:
        seg_to_lines = _map_lines_to_segments_by_duration(rough_segments, official_lines)
    if job_log:
        method_label = "text-aware" if USE_TEXT_AWARE_SEGMENT_MAPPING else "duration"
        job_log.info(
            f"[assign] Segment mapping method: {method_label}",
            stage="transcribe",
        )

    # Debug instrumentation — observability only, no effect on alignment
    from alignment.debug import AlignmentDebugger
    _dbg = AlignmentDebugger(len(rough_segments), len(official_lines))

    # ── Collapse detection ────────────────────────────────────────────────────
    # Warn when a segment window is too short for its assigned lines.
    # These segments will use silence detection (not proportional distribution)
    # in the fallback path below.
    for i, seg in enumerate(rough_segments):
        lines_here = seg_to_lines.get(i, [])
        if not lines_here:
            continue
        dur     = seg.get('end', 0.0) - seg.get('start', 0.0)
        dur_per = dur / len(lines_here)
        if dur_per < _MIN_DUR_PER_LINE and job_log:
            job_log.warning(
                f"[align] Collapsed segment {i}: {len(lines_here)} lines in "
                f"{dur:.2f}s ({dur_per:.2f}s/line) at {seg.get('start', 0.0):.1f}s",
                stage="transcribe",
            )

    # ── 2. WhisperX acoustic alignment ───────────────────────────────────────
    new_segs:     list[dict]                  = []
    seg_line_map: list[Optional[list[str]]]   = []

    for i, seg in enumerate(rough_segments):
        lines = seg_to_lines.get(i)
        new_segs.append({
            'text':  ' '.join(lines) if lines else seg.get('text', ''),
            'start': seg.get('start', 0.0),
            'end':   seg.get('end',   0.0),
        })
        seg_line_map.append(lines)

    try:
        audio       = whisperx.load_audio(str(audio_path))
        align_model, align_meta = whisperx.load_align_model(
            language_code=WHISPER_LANGUAGE, device=device,
        )
        aligned      = whisperx.align(
            new_segs, align_model, align_meta, audio, device,
            return_char_alignments=True,
        )
        del align_model
        _free_vram()
        aligned_segs = aligned.get('segments', new_segs)
    except Exception as exc:
        if job_log:
            job_log.warning(f"[align] WhisperX alignment failed, using proportional: {exc}", stage="transcribe")
        aligned_segs = new_segs

    # ── 3. Reconcile aligned segments → rough segments by time overlap ───────
    if len(aligned_segs) != len(rough_segments):
        if job_log:
            job_log.info(
                f"[align] WhisperX returned {len(aligned_segs)} segments "
                f"(expected {len(rough_segments)}) — reconciling by time overlap "
                f"to preserve char-level acoustic data",
                stage="transcribe",
            )
        seg_map, unmatched = _reconcile_by_overlap(rough_segments, aligned_segs)
        if unmatched and job_log:
            job_log.warning(
                f"[align] {len(unmatched)} aligned segment(s) fell outside all "
                f"rough-segment windows and were discarded",
                stage="transcribe",
            )
    else:
        # 1:1 index mapping (the count-matched path)
        seg_map   = {i: [aseg] for i, aseg in enumerate(aligned_segs)}
        unmatched = []

    # ── 4. Build per-line results ─────────────────────────────────────────────
    timing_sources: dict[str, int] = {
        'char_boundary': 0,
        'word_boundary': 0,
        'silence_split': 0,
        'proportional':  0,
    }
    final: list[dict] = []
    _anchor_offsets: list[float] = []   # per-segment applied offset (ms), 0 = skipped

    for i, (orig_seg, line_texts) in enumerate(zip(rough_segments, seg_line_map)):
        matched   = seg_map.get(i, [])
        seg_start = orig_seg.get('start', 0.0)
        seg_end   = orig_seg.get('end',   0.0)

        if not line_texts:
            final.extend(matched if matched else [orig_seg])
            continue

        # Collect char/word data from all aligned sub-segments for this window
        chars = _collect_chars(matched)
        words = _collect_words(matched)

        # Raw WhisperX dump — shows exactly what CTC alignment produced
        if matched and job_log:
            for _mi, _ms in enumerate(matched):
                _log_wx_raw(i, _ms, job_log)

        # ── Debug: record segment baseline ───────────────────────────────────
        _dbg.record_segment(i, seg_start, seg_end, matched, line_texts or [], chars, words)

        # Use the tightest acoustic time bounds, clamped to rough ± tolerance
        tolerance = 2.0
        if matched:
            eff_start = max(seg_start - tolerance, matched[0].get('start', seg_start))
            eff_end   = min(seg_end   + tolerance, matched[-1].get('end',   seg_end))
        else:
            eff_start, eff_end = seg_start, seg_end

        # ── VAD anchor: remove per-segment CTC lag ───────────────────────────
        # Shifts word timestamps so the first word starts at the VAD boundary.
        # Skipped when the offset is large enough to indicate an instrumental fill.
        words, _anchor_ms = _apply_vad_anchor(words, seg_start, i, job_log)
        _anchor_offsets.append(_anchor_ms)
        anchored_start = seg_start if _anchor_ms > 0.0 else eff_start
        eff_seg = {'start': anchored_start, 'end': eff_end, 'text': '', 'words': words}

        # ── Timing priority: word-gap → char timestamps → proportional ─────────
        # Character ratio is NEVER used for timing; only word/char timestamps.
        candidates: list[dict] = []
        source     = 'proportional'

        if len(line_texts) > 1:
            if words:
                # ── Detect character-level tokenization (Japanese) ─────────────
                # When WhisperX tokenises at char level, end=next_start everywhere
                # → all inter-token gaps are 0 ms.  Gap-based splitting is useless
                # in that regime; jump straight to char progression.
                _end_gaps = [
                    (float(words[k+1].get('start', 0)) - float(words[k].get('end', 0))) * 1000
                    for k in range(min(8, len(words) - 1))
                    if words[k].get('end') is not None and words[k+1].get('start') is not None
                ]
                _is_char_level = (
                    len(_end_gaps) >= 3 and
                    max(_end_gaps) < _ZERO_GAP_THRESHOLD_MS
                )

                if _is_char_level:
                    # PRIMARY (char-level): mora-weighted char START progression
                    if job_log:
                        job_log.info(
                            f"[char_boundary] Seg{i}: char-level tokenisation "
                            f"(max_end_gap={max(_end_gaps):.1f}ms, {len(words)} tokens) "
                            f"— using mora-weighted progression",
                            stage="transcribe",
                        )
                    candidates = _split_by_char_progression(
                        words, line_texts, eff_seg, i, job_log
                    )
                    source = 'char_boundary'

                    # Safeguard on char result — try relaxed limit before proportional
                    bad, reason = _is_pathological_split(candidates)
                    if bad:
                        # Intermediate check: accept if ratio is only moderately
                        # extreme (< relaxed limit) AND min_dur is above the display
                        # floor. Char timestamps are acoustically real even when the
                        # ratio is high — proportional would discard that information.
                        bad_relaxed, _ = _is_pathological_split(
                            candidates, max_ratio=_SAFEGUARD_MAX_RATIO_RELAXED
                        )
                        if not bad_relaxed:
                            if job_log:
                                job_log.info(
                                    f"[fallback] Seg{i}: char_progression "
                                    f"char_progression=borderline({reason}) "
                                    f"fallback=char_boundary_relaxed "
                                    f"reason=ratio_within_relaxed_limit({_SAFEGUARD_MAX_RATIO_RELAXED}x)",
                                    stage="transcribe",
                                )
                            source = 'char_boundary'
                        else:
                            if job_log:
                                job_log.warning(
                                    f"[fallback] Seg{i}: char_progression=rejected({reason}) "
                                    f"fallback=proportional "
                                    f"reason=ratio_exceeded_relaxed_limit",
                                    stage="transcribe",
                                )
                            candidates = _proportional_line(orig_seg, line_texts)
                            source     = 'proportional'
                    else:
                        _log_token_map(candidates, i, job_log)
                        _compare_splits(
                            candidates, _proportional_line(orig_seg, line_texts), i, job_log
                        )

                else:
                    # PRIMARY (word-level): gap-based boundary detection
                    candidates = _split_by_word_gaps(
                        words, line_texts, eff_seg,
                        job_log=job_log, seg_id=i,
                    )
                    source = 'word_boundary'

                    # Safeguard: reject pathological splits
                    bad, reason = _is_pathological_split(candidates)
                    if bad:
                        if job_log:
                            job_log.warning(
                                f"[safeguard] Seg{i}: word-gap split rejected — {reason} "
                                f"— trying char_boundary",
                                stage="transcribe",
                            )
                        # Try char progression before giving up on acoustic timing
                        char_cands = _split_by_char_progression(
                            words, line_texts, eff_seg, i, job_log
                        )
                        char_bad, char_reason = _is_pathological_split(char_cands)
                        if not char_bad:
                            candidates = char_cands
                            source     = 'char_boundary'
                            if job_log:
                                job_log.info(
                                    f"[char_boundary] Seg{i}: accepted after word-gap rejection",
                                    stage="transcribe",
                                )
                            _log_token_map(candidates, i, job_log)
                            _compare_splits(
                                candidates, _proportional_line(orig_seg, line_texts), i, job_log
                            )
                        else:
                            if job_log:
                                job_log.warning(
                                    f"[safeguard] Seg{i}: char progression also rejected "
                                    f"({char_reason}) — falling back to proportional",
                                    stage="transcribe",
                                )
                            candidates = _proportional_line(orig_seg, line_texts)
                            source     = 'proportional'
            elif chars:
                # SECONDARY: char-level CTC timestamps (no word data available)
                char_cands = _boundary_from_chars(chars, line_texts, seg_start, seg_end)
                if char_cands:
                    candidates = char_cands
                    source     = 'char_boundary'
                else:
                    candidates = _proportional_line(orig_seg, line_texts)
                    source     = 'proportional'
            else:
                # LAST RESORT: no acoustic data — proportional only
                candidates = _proportional_line(orig_seg, line_texts)
                source     = 'proportional'
        else:
            # Single-line segment — use aligned segment bounds
            cand   = {**eff_seg, 'text': line_texts[0], 'words': words}
            source = 'word_boundary' if words else ('char_boundary' if chars else 'proportional')
            candidates = [cand]

        # ── Validate: reject candidates outside the rough-segment window ────
        valid = candidates and all(
            seg_start - tolerance <= c.get('start', 0.0) <= seg_end + tolerance
            for c in candidates
            if c.get('start', 0.0) > 0
        )

        if valid:
            # ── Intra-segment tail guard + drift correction ────────────────────
            # Small gaps (< _MAX_LINE_EXTENSION_MS) between the last acoustic
            # line-end and seg_end are normal CTC lag — extend to cover them.
            # Large gaps are instrumental tails — capping prevents a 26s lyric.
            if candidates and source in ('word_boundary', 'char_boundary'):
                last_end     = candidates[-1].get('end', 0.0)
                seg_remain   = seg_end - last_end
                remain_ms    = seg_remain * 1000

                if remain_ms > _DRIFT_CORRECT_THRESH * 1000:
                    if remain_ms <= _MAX_LINE_EXTENSION_MS:
                        # Small tail — safe to extend to seg_end
                        candidates[-1]['end'] = round(seg_end, 3)
                        if job_log:
                            job_log.info(
                                f"[drift] Seg{i}: +{remain_ms:.0f}ms correction "
                                f"({last_end:.3f}s → {seg_end:.3f}s)",
                                stage="transcribe",
                            )
                    else:
                        # Large tail — check vocal energy before extending
                        asr_path = audio_path.parent / 'vocals_asr.wav'
                        probe_path = asr_path if asr_path.exists() else audio_path
                        energy = _vocal_energy_in_window(
                            probe_path, last_end,
                            min(last_end + 1.0, seg_end)   # probe first 1s of tail
                        )
                        tail_type = (
                            'instrumental_tail' if remain_ms > _INSTRUMENTAL_TAIL_MS
                            else 'transition_gap'
                        )

                        if energy < _VOCAL_ENERGY_THRESHOLD:
                            # Silence/instrumental — cap extension hard
                            capped_end = round(last_end + _MAX_LINE_EXTENSION_MS / 1000, 3)
                            candidates[-1]['end'] = capped_end
                            if job_log:
                                job_log.info(
                                    f"[tail_guard] Seg{i}: {tail_type} "
                                    f"+{remain_ms:.0f}ms, energy={energy:.4f} < threshold "
                                    f"— capped at +{_MAX_LINE_EXTENSION_MS}ms "
                                    f"({last_end:.3f}s → {capped_end:.3f}s)",
                                    stage="transcribe",
                                )
                        else:
                            # Vocal energy present — allow up to cap
                            capped_end = round(
                                min(seg_end, last_end + _MAX_LINE_EXTENSION_MS / 1000), 3
                            )
                            candidates[-1]['end'] = capped_end
                            if job_log:
                                job_log.info(
                                    f"[tail_guard] Seg{i}: {tail_type} "
                                    f"+{remain_ms:.0f}ms, energy={energy:.4f} (vocal present) "
                                    f"— capped at +{_MAX_LINE_EXTENSION_MS}ms",
                                    stage="transcribe",
                                )

                        if job_log:
                            job_log.info(
                                f"[instrumental_tail] Seg{i}: "
                                f"tail={remain_ms:.0f}ms type={tail_type} "
                                f"energy={energy:.4f}",
                                stage="transcribe",
                            )

            # ── Log final chosen boundaries ───────────────────────────────────
            if job_log:
                for li, c in enumerate(candidates):
                    dur  = c.get('end', 0.0) - c.get('start', 0.0)
                    flag = ' ⚑SHORT' if dur < _MIN_LINE_DISPLAY_S else ''
                    job_log.info(
                        f"[boundary] Seg{i} line{li}: "
                        f"{c.get('start', 0):.3f}–{c.get('end', 0):.3f}s "
                        f"dur={dur:.3f}s src={source}{flag}  "
                        f"'{c.get('text', '')[:32]}'",
                        stage="transcribe",
                    )

            _segment_quality(candidates, source, i, job_log)

            for c in candidates:
                c['_timing_source'] = source
            final.extend(candidates)
            timing_sources[source] += len(candidates)
            # Debug: record each accepted line
            counts = [max(1, _char_count(t)) for t in (line_texts or [])]
            total_c = max(1, sum(counts))
            total_w = max(1, len(words))
            char_cum = 0
            for li, (cand, cnt) in enumerate(zip(candidates, counts)):
                char_ratio = char_cum / total_c
                word_ratio = round(cnt / total_c * total_w / total_w, 4)
                sil = _dbg.probe_silence(audio_path, cand.get('start', seg_start))
                _dbg.record_line(
                    len(final) - len(candidates) + li, cand, i,
                    seg_start, seg_end, char_ratio, word_ratio,
                    source, cand.get('words', []), sil,
                )
                char_cum += cnt
        else:
            if job_log:
                job_log.warning(
                    f"[align] Timestamps outside window [{seg_start:.1f}-{seg_end:.1f}s] "
                    f"for '{line_texts[0][:20]}' — trying silence split",
                    stage="transcribe",
                )
            silence_lines = _silence_split_fallback(orig_seg, line_texts, audio_path, job_log)
            # Silence split succeeds if it found distinct boundaries
            n_unique = len({c['start'] for c in silence_lines})
            sil_src  = 'silence_split' if (n_unique > 1 or len(silence_lines) == 1) else 'proportional'
            for c in silence_lines:
                c['_timing_source'] = sil_src

            if job_log:
                for li, c in enumerate(silence_lines):
                    dur  = c.get('end', 0.0) - c.get('start', 0.0)
                    flag = ' ⚑SHORT' if dur < _MIN_LINE_DISPLAY_S else ''
                    job_log.info(
                        f"[boundary] Seg{i} line{li}: "
                        f"{c.get('start', 0):.3f}–{c.get('end', 0):.3f}s "
                        f"dur={dur:.3f}s src={sil_src}{flag}  "
                        f"'{c.get('text', '')[:32]}'",
                        stage="transcribe",
                    )

            final.extend(silence_lines)
            timing_sources[sil_src] += len(silence_lines)
            # Debug: record fallback lines
            counts = [max(1, _char_count(t)) for t in (line_texts or [])]
            total_c = max(1, sum(counts))
            char_cum = 0
            for li, (cand, cnt) in enumerate(zip(silence_lines, counts)):
                sil = _dbg.probe_silence(audio_path, cand.get('start', seg_start))
                _dbg.record_line(
                    len(final) - len(silence_lines) + li, cand, i,
                    seg_start, seg_end, char_cum / total_c, cnt / total_c,
                    sil_src, cand.get('words', []), sil,
                )
                char_cum += cnt

    # ── Anchor summary ────────────────────────────────────────────────────────
    n_anchored = sum(1 for o in _anchor_offsets if o > 0)
    if n_anchored > 0 and job_log:
        applied = [o for o in _anchor_offsets if o > 0]
        job_log.info(
            f"[anchor] VAD anchor applied to {n_anchored}/{len(rough_segments)} segments, "
            f"mean_offset={sum(applied)/len(applied):.0f}ms "
            f"min={min(applied):.0f}ms max={max(applied):.0f}ms",
            stage="transcribe",
        )

    # ── 5a. Drift curve (before gap-bridging adjusts end times) ─────────────
    _log_drift_curve(final, rough_segments, job_log)

    # ── 5b. Boundary audit ───────────────────────────────────────────────────
    _audit_boundaries(final, job_log)

    # ── 5. Bridge inter-line gaps ─────────────────────────────────────────────
    for i in range(len(final) - 1):
        cur_end    = final[i].get('end',   0.0)
        next_start = final[i + 1].get('start', 0.0)
        gap_ms     = (next_start - cur_end) * 1000
        if gap_ms <= 0:
            continue
        if gap_ms <= _MAX_LINE_EXTENSION_MS:
            # Small gap — bridge normally for smooth karaoke flow
            final[i]['end'] = round(next_start - 0.05, 3)
        else:
            # Large gap (instrumental section) — cap extension, do not absorb
            capped = round(cur_end + _MAX_LINE_EXTENSION_MS / 1000, 3)
            final[i]['end'] = min(capped, round(next_start - 0.05, 3))
            if job_log:
                job_log.info(
                    f"[tail_guard] line{i}: gap={gap_ms:.0f}ms > {_MAX_LINE_EXTENSION_MS}ms "
                    f"— capped bridge at +{_MAX_LINE_EXTENSION_MS}ms "
                    f"(acoustic {cur_end:.3f}s → capped {final[i]['end']:.3f}s, "
                    f"next starts {next_start:.3f}s)",
                    stage="transcribe",
                )

    # ── 6. Summary metrics ────────────────────────────────────────────────────
    n_total    = max(1, len(final))
    n_acoustic = timing_sources['char_boundary'] + timing_sources['word_boundary']
    n_prop     = timing_sources['proportional']
    n_short    = sum(1 for ln in final
                     if ln.get('end', 0.0) - ln.get('start', 0.0) < _MIN_DUR_PER_LINE)

    if job_log:
        job_log.info(
            f"[align] Alignment complete: {n_total} lines | "
            f"char={timing_sources['char_boundary']} "
            f"word={timing_sources['word_boundary']} "
            f"silence={timing_sources['silence_split']} "
            f"proportional={n_prop} "
            f"({100 * n_acoustic // n_total}% acoustic)"
            + (f" | {n_short} short (<{_MIN_DUR_PER_LINE}s)" if n_short else ""),
            stage="transcribe",
        )
        if n_prop > n_total * 0.20 and n_acoustic > 0:
            job_log.warning(
                f"[align] {n_prop}/{n_total} lines ({100*n_prop//n_total}%) used "
                f"proportional fallback — char-level alignment may be incomplete",
                stage="transcribe",
            )
        if n_acoustic == 0:
            job_log.warning(
                "[align] No acoustic boundaries found — all lines are proportional estimates",
                stage="transcribe",
            )

    # ── Karaoke health check ─────────────────────────────────────────────────
    _karaoke_health_check(final, job_log)

    # ── Sanity validation ─────────────────────────────────────────────────────
    _sanity_validate(final, job_log)

    # ── Save debug report ─────────────────────────────────────────────────────
    try:
        _dbg.save(audio_path.parent)
    except Exception as _de:
        logger.debug("alignment_debug.json write failed: %s", _de)

    return final, "whisperx_official"


# ── Mora-weighted char-progression splitter ───────────────────────────────────
#
# WhisperX for Japanese produces CHARACTER-LEVEL tokens where:
#   - end times = next token's start time  →  0 ms inter-token gaps
#   - but start times represent real acoustic onsets
#
# Gap-based splitting is useless (all gaps 0 ms) but the START progression
# is acoustically meaningful.  We map each lyric line's mora weight to a
# fractional position in the character sequence and read the timestamp there.

_SMALL_KANA = frozenset('ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ')
_ZERO_GAP_THRESHOLD_MS = 40   # max end→next-start gap to classify as char-level


def _mora_count(text: str) -> int:
    """
    Estimate Japanese mora count for timing weight.

    Moras are the primary timing unit in Japanese speech/singing.
    Better weight than raw UTF-8 length when distributing line durations.

      - Regular kana              → 1 mora each
      - Small kana (ぁ etc.)      → 0 (they modify the preceding mora)
      - Long vowel ー / wave 〜   → 1 mora
      - Kanji                     → 2 moras (rough average)
      - ASCII letters             → 1 per character (rough)
      - Punctuation / whitespace  → 0
    """
    count = 0
    for ch in unicodedata.normalize('NFKC', text):
        cp = ord(ch)
        if ch in _SMALL_KANA:
            pass
        elif 0x3041 <= cp <= 0x3096 or 0x30A1 <= cp <= 0x30F6:   # hiragana / katakana
            count += 1
        elif ch in ('ー', '〜'):                                   # long vowel
            count += 1
        elif 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:   # kanji
            count += 2
        elif ch.isascii() and ch.isalpha():
            count += 1
    return max(1, count)


def _split_by_char_progression(
    words:      list[dict],
    line_texts: list[str],
    parent_seg: dict,
    seg_id:     int = -1,
    job_log=None,
) -> list[dict]:
    """
    Split lyric lines using cumulative character START timestamp progression.

    Algorithm:
      1. Compute mora weight for each lyric line.
      2. Map cumulative mora fraction → index in the character sequence.
      3. Read the char START timestamp at that index as the line boundary.

    End time of each line = START time of the next line's first char (not
    the current line's last char's END time, which equals the next char's
    START anyway in char-level output).

    Works for continuous Japanese singing where gap-based splitting fails.
    """
    n = len(line_texts)
    if not words or n == 0:
        return _proportional_line(parent_seg, line_texts)

    if n == 1:
        return [{
            'text':  line_texts[0],
            'start': round(float(words[0].get('start') or parent_seg.get('start', 0.0)), 3),
            'end':   round(float(words[-1].get('end')  or parent_seg.get('end',   0.0)), 3),
            'words': words,
        }]

    total_chars = len(words)
    mora_counts = [max(1, _mora_count(t)) for t in line_texts]
    total_moras = sum(mora_counts)

    # Cumulative mora fraction → start index in char sequence
    start_indices: list[int] = []
    cumul = 0
    for mc in mora_counts:
        idx = min(int(cumul / total_moras * total_chars), total_chars - 1)
        start_indices.append(idx)
        cumul += mc

    # Mean char spacing for diagnostics
    ts = [float(w.get('start', 0)) for w in words if w.get('start') is not None]
    mean_spacing_ms = (
        (ts[-1] - ts[0]) / max(1, len(ts) - 1) * 1000 if len(ts) > 1 else 0
    )

    result: list[dict] = []
    for li, (text, si) in enumerate(zip(line_texts, start_indices)):
        ei      = start_indices[li + 1] if li + 1 < n else total_chars
        grp     = words[si:ei]

        start_t = float(words[si].get('start') or parent_seg.get('start', 0.0))
        # End = start of next line's first char (acoustically more accurate than
        # current last char's end, which is just next char's start anyway)
        if li + 1 < n and start_indices[li + 1] < total_chars:
            end_t = float(
                words[start_indices[li + 1]].get('start')
                or parent_seg.get('end', 0.0)
            )
        else:
            end_t = float(words[-1].get('end') or parent_seg.get('end', 0.0))

        result.append({
            'text':  text,
            'start': round(start_t, 3),
            'end':   round(end_t,   3),
            'words': grp,
        })

        # Confidence for char-boundary: based on chars-per-line density
        # More chars per line = more stable boundary estimation
        b_conf  = min(0.80, 0.30 + len(grp) / max(total_chars, 1) * 2.0)
        b_label = 'char_dense' if b_conf >= 0.60 else 'char_sparse'

        result[-1]['_boundary_conf']       = round(b_conf, 2)
        result[-1]['_boundary_conf_label'] = b_label

        if job_log:
            dur_flag = ' ⚑SHORT' if (end_t - start_t) < _MIN_LINE_DISPLAY_S else ''
            job_log.info(
                f"[char_boundary] Seg{seg_id} L{li}: "
                f"{start_t:.3f}–{end_t:.3f}s "
                f"dur={round(end_t-start_t,3):.3f}s "
                f"chars={len(grp)} mora={mora_counts[li]} idx={si} "
                f"bconf={b_conf:.2f}({b_label}){dur_flag}  "
                f"'{text[:30]}'",
                stage="transcribe",
            )

    if job_log:
        job_log.info(
            f"[char_boundary] Seg{seg_id}: "
            f"mean_char_spacing={mean_spacing_ms:.0f}ms  "
            f"mora_weights={mora_counts}  total_chars={total_chars}",
            stage="transcribe",
        )

    return result


# ── Pathological split detection ──────────────────────────────────────────────

_SAFEGUARD_MIN_DUR_S      = 0.40   # any line shorter than this triggers rejection
_SAFEGUARD_MAX_RATIO      = 8.0    # longest/shortest duration ratio ceiling
_SAFEGUARD_MAX_RATIO_RELAXED = 12.0  # relaxed limit used as intermediate before proportional
_SAFEGUARD_MAX_1WORD_FRAC = 0.50   # max fraction of lines with ≤1 word token


def _is_pathological_split(
    candidates: list[dict],
    max_ratio: float | None = None,
) -> tuple[bool, str]:
    """
    Return (True, reason) when acoustic word-gap splitting produced unusable output.

    Three independent checks — any one failing rejects the split:
      1. Ultra-short line  — any duration < _SAFEGUARD_MIN_DUR_S
      2. Extreme ratio     — max_dur / min_dur > _SAFEGUARD_MAX_RATIO
      3. Single-word flood — > 50 % of lines have ≤ 1 word token

    A rejected split falls back to proportional distribution so the song
    never gets 0.02 s lines or one line swallowing 26 seconds.

    max_ratio: override the ratio ceiling (used for relaxed intermediate check).
    """
    if len(candidates) <= 1:
        return False, ""

    effective_max_ratio = max_ratio if max_ratio is not None else _SAFEGUARD_MAX_RATIO

    durations = [max(0.0, c.get('end', 0.0) - c.get('start', 0.0)) for c in candidates]
    min_dur   = min(durations)
    max_dur   = max(durations)

    if min_dur < _SAFEGUARD_MIN_DUR_S:
        return True, f"ultra-short line: min_dur={min_dur:.3f}s"

    if min_dur > 0 and max_dur / min_dur > effective_max_ratio:
        return True, (
            f"extreme ratio: {max_dur:.2f}s/{min_dur:.2f}s "
            f"= {max_dur/min_dur:.1f}× (limit {effective_max_ratio}×)"
        )

    n_sparse = sum(1 for c in candidates if len(c.get('words', [])) <= 1)
    frac     = n_sparse / len(candidates)
    if frac > _SAFEGUARD_MAX_1WORD_FRAC:
        return True, (
            f"too many 1-word lines: {n_sparse}/{len(candidates)} "
            f"= {frac:.0%} (limit {_SAFEGUARD_MAX_1WORD_FRAC:.0%})"
        )

    return False, ""


# ── Vocal energy probe ────────────────────────────────────────────────────────

def _vocal_energy_in_window(audio_path: Path, start: float, end: float) -> float:
    """
    Return mean RMS energy in [start, end] of the ASR audio (0–1 scale).
    Values below _VOCAL_ENERGY_THRESHOLD indicate silence / instrumental.
    Returns 0.5 (conservative) if audio cannot be read.
    """
    if end <= start:
        return 0.0
    try:
        import soundfile as sf
        import numpy as np
        info     = sf.info(str(audio_path))
        sr       = info.samplerate
        s_frame  = int(start * sr)
        n_frames = max(1, int((end - start) * sr))
        audio, _ = sf.read(str(audio_path), start=s_frame, frames=n_frames, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return float(np.sqrt(np.mean(audio ** 2) + 1e-9))
    except Exception:
        return 0.5   # can't read → conservative, don't block extension


# ── Deep observability helpers ────────────────────────────────────────────────

def _split_quality_metrics(candidates: list[dict]) -> dict:
    """Pure-function quality metrics for a list of timed line dicts."""
    if not candidates:
        return {'n': 0}
    durs    = [max(0.0, c.get('end', 0.0) - c.get('start', 0.0)) for c in candidates]
    n       = len(durs)
    min_d   = min(durs);  max_d = max(durs)
    mean_d  = sum(durs) / n
    mad     = sum(abs(d - mean_d) for d in durs) / n
    ratio   = max_d / min_d if min_d > 0 else float('inf')
    n_short = sum(1 for d in durs if d < _SAFEGUARD_MIN_DUR_S)
    wcs     = [len(c.get('words', [])) for c in candidates]
    return {
        'n': n,
        'min':   round(min_d,  3),
        'max':   round(max_d,  3),
        'mean':  round(mean_d, 3),
        'mad':   round(mad,    3),
        'ratio': round(ratio, 1) if ratio != float('inf') else float('inf'),
        'n_short':      n_short,
        'n_zero_words': sum(1 for w in wcs if w == 0),
        'word_counts':  wcs,
    }


def _boundary_conf_score(gap_ms: Optional[float], method: str) -> tuple[float, str]:
    """Return (confidence 0–1, label) for a single line boundary."""
    if method in ('proportional', 'char_weighted', 'silence_split'):
        return 0.10, 'no_acoustic'
    if method == 'char_boundary':
        return 0.50, 'char'
    if gap_ms is None:
        return 0.20, 'unknown'
    if gap_ms >= _WORD_GAP_SPLIT_MS:    # default 120 ms
        return 0.85, 'strong_gap'
    if gap_ms >= 50:
        return 0.60, 'moderate_gap'
    if gap_ms >= 20:
        return 0.35, 'weak_gap'
    return 0.15, 'micro_gap'


def _compare_splits(
    acoustic:     list[dict],
    proportional: list[dict],
    seg_id:       int,
    job_log=None,
) -> None:
    """[compare] Side-by-side quality metrics for acoustic vs proportional splits."""
    if not job_log:
        return
    am      = _split_quality_metrics(acoustic)
    pm      = _split_quality_metrics(proportional)
    a_ratio = am.get('ratio', float('inf'))
    p_ratio = pm.get('ratio', 1.0)
    verdict = (
        'proportional_superior'
        if (isinstance(a_ratio, float) and a_ratio > max(p_ratio * 2, 8))
        else 'acoustic_ok'
    )
    job_log.info(
        f"[compare] Seg{seg_id}  "
        f"acoustic ratio={a_ratio}x mad={am.get('mad',0):.3f}s short={am.get('n_short',0)} | "
        f"proportional ratio={p_ratio}x mad={pm.get('mad',0):.3f}s "
        f"→ {verdict}",
        stage="transcribe",
    )


def _segment_quality(
    candidates: list[dict],
    method:     str,
    seg_id:     int,
    job_log=None,
) -> float:
    """[quality] Compute and log a quality score (0–1) for a segment's split."""
    m     = _split_quality_metrics(candidates)
    score = 1.0
    flags: list[str] = []

    if m.get('n_short', 0) > 0:
        score -= 0.30
        flags.append(f"{m['n_short']} ultra-short")

    ratio = m.get('ratio', 1.0)
    if isinstance(ratio, float) and ratio > _SAFEGUARD_MAX_RATIO:
        score -= min(0.40, (ratio - _SAFEGUARD_MAX_RATIO) / 50)
        flags.append(f"ratio={ratio:.0f}x")

    if method in ('proportional', 'char_weighted'):
        score -= 0.15
        flags.append(f"method={method}")

    n_zero = m.get('n_zero_words', 0)
    if m.get('n', 1) and n_zero > m['n'] * 0.3:
        score -= 0.20
        flags.append(f"{n_zero} zero-word lines")

    score = round(max(0.0, min(1.0, score)), 3)

    if job_log:
        flag_str = ' | '.join(flags) if flags else 'ok'
        level    = 'warning' if score < 0.40 else 'info'
        getattr(job_log, level)(
            f"[quality] Seg{seg_id} score={score:.2f}"
            + (' ⚑BAD' if score < 0.40 else '')
            + f"  mad={m.get('mad',0):.3f}s ratio={ratio}x"
            + f"  words={m.get('word_counts',[])}  {flag_str}",
            stage="transcribe",
        )
    return score


def _log_token_map(result: list[dict], seg_id: int, job_log=None) -> None:
    """[token_map] Log exact word-token assignments per lyric line."""
    if not job_log:
        return
    job_log.info(f"[token_map] Seg{seg_id}:", stage="transcribe")
    for li, line in enumerate(result):
        words  = line.get('words', [])
        tokens = [w.get('word', '?') for w in words]
        t0     = words[0].get('start') if words else None
        t1     = words[-1].get('end')  if words else None
        span   = f"[{t0:.3f}–{t1:.3f}s]" if t0 is not None and t1 is not None else "[no_ts]"
        conf   = line.get('_boundary_conf')
        conf_s = f" bconf={conf:.2f}" if conf is not None else ""
        job_log.info(
            f"[token_map]   L{li} {span}{conf_s} ← {tokens[:12]}"
            + (" …" if len(tokens) > 12 else ""),
            stage="transcribe",
        )


def _log_wx_raw(seg_idx: int, seg: dict, job_log=None) -> None:
    """[wx_raw] Log raw WhisperX segment data before post-processing (first+last 5 words)."""
    if not job_log:
        return
    words    = seg.get('words', [])
    n_words  = len(words)
    head     = words[:5]
    tail     = words[-2:] if n_words > 7 else []
    fmt      = lambda w: f"{w.get('word','?')}@{w.get('start',0):.2f}s"  # noqa: E731
    parts    = [fmt(w) for w in head]
    if tail:
        parts += ['…'] + [fmt(w) for w in tail]
    job_log.info(
        f"[wx_raw] Seg{seg_idx} {seg.get('start',0):.3f}–{seg.get('end',0):.3f}s "
        f"({n_words} words): [{', '.join(parts)}]",
        stage="transcribe",
    )


def _log_drift_curve(
    final:          list[dict],
    rough_segments: list[dict],
    job_log=None,
) -> None:
    """[drift_curve] Per-segment drift: actual first-line-start vs rough-seg-start."""
    if not job_log or not final or not rough_segments:
        return
    job_log.info("[drift_curve] Seg  rough_start → first_line_start  drift  cumulative",
                 stage="transcribe")
    cumul = 0.0
    for si, rough in enumerate(rough_segments):
        rs = rough.get('start', 0.0)
        re = rough.get('end',   0.0)
        # Find first final line whose start falls in or near this segment
        first = next(
            (l for l in final if rs - 1.0 <= l.get('start', 0.0) <= re + 1.0),
            None,
        )
        if first is None:
            continue
        drift_ms = (first.get('start', rs) - rs) * 1000
        cumul   += drift_ms
        flag     = '  ⚑LARGE' if abs(drift_ms) > 500 else ''
        job_log.info(
            f"[drift_curve] Seg{si:02d}  {rs:.3f}s → {first.get('start',rs):.3f}s  "
            f"{drift_ms:+.0f}ms  cumul={cumul:+.0f}ms{flag}",
            stage="transcribe",
        )


def _karaoke_health_check(lines: list[dict], job_log) -> None:
    """
    [karaoke_health] End-to-end sanity check for karaoke playback quality.

    Emits a scored summary with per-metric flags so the logs tell you at a
    glance whether the output is ready for display or needs investigation.
    Does NOT modify lines — diagnostic only.
    """
    if not job_log or not lines:
        return

    n = len(lines)
    durs = [max(0.0, l.get('end', 0) - l.get('start', 0)) for l in lines]
    gaps = [
        lines[i].get('start', 0) - lines[i - 1].get('end', 0)
        for i in range(1, n)
    ]

    # ── Per-metric counts ────────────────────────────────────────────────────
    n_short      = sum(1 for d in durs if d < _MIN_LINE_DISPLAY_S)
    n_very_long  = sum(1 for d in durs if d > 15.0)
    n_long       = sum(1 for d in durs if 8.0 < d <= 15.0)
    n_big_gap    = sum(1 for g in gaps if g > 5.0)
    n_overlap    = sum(1 for g in gaps if g < -0.01)
    max_dur      = max(durs) if durs else 0.0
    total_vocal  = sum(durs)

    # Lines whose duration exceeds the cap (tail guard should have caught these)
    n_tail_viol  = sum(
        1 for i in range(n - 1)
        if (lines[i].get('end', 0) - lines[i].get('_acoustic_end', lines[i].get('end', 0))) * 1000
           > _MAX_LINE_EXTENSION_MS + 50  # 50ms tolerance
    )

    # Timing source breakdown
    sources: dict[str, int] = {}
    for ln in lines:
        src = ln.get('_timing_source', 'unknown')
        sources[src] = sources.get(src, 0) + 1
    acoustic = sources.get('char_boundary', 0) + sources.get('word_boundary', 0)
    acoustic_pct = int(100 * acoustic / n) if n else 0

    # ── Score (0–100) ────────────────────────────────────────────────────────
    score = 100
    flags: list[str] = []

    if n_short:
        score -= min(20, n_short * 4)
        flags.append(f"{n_short} short(<{_MIN_LINE_DISPLAY_S}s)")
    if n_very_long:
        score -= min(30, n_very_long * 15)
        flags.append(f"{n_very_long} very-long(>15s)")
    if n_long:
        score -= min(10, n_long * 3)
        flags.append(f"{n_long} long(>8s)")
    if n_big_gap:
        score -= min(10, n_big_gap * 3)
        flags.append(f"{n_big_gap} big-gap(>5s)")
    if n_overlap:
        score -= min(20, n_overlap * 5)
        flags.append(f"{n_overlap} overlaps")
    if acoustic_pct < 60:
        score -= (60 - acoustic_pct) // 3
        flags.append(f"acoustic_pct={acoustic_pct}%")

    score = max(0, score)
    grade = "EXCELLENT" if score >= 90 else "GOOD" if score >= 75 else "FAIR" if score >= 55 else "POOR"

    job_log.info(
        f"[karaoke_health] score={score}/100 ({grade}) | "
        f"lines={n} acoustic={acoustic_pct}% "
        f"max_dur={max_dur:.1f}s total_vocal={total_vocal:.1f}s"
        + (f" | flags: {', '.join(flags)}" if flags else " | no issues"),
        stage="align",
    )

    # Per-line detail for any very-long lines
    for i, (ln, d) in enumerate(zip(lines, durs)):
        if d > 8.0:
            src  = ln.get('_timing_source', '?')
            text = (ln.get('text') or '')[:30]
            job_log.info(
                f"[karaoke_health] long line{i}: {ln.get('start', 0):.3f}–{ln.get('end', 0):.3f}s "
                f"dur={d:.1f}s src={src} '{text}'",
                stage="align",
            )


def _sanity_validate(lines: list[dict], job_log) -> bool:
    """
    [sanity] Final validation pass before writing aligned_lines.json.
    Logs PASS/FAIL with specifics. Does NOT block writing — diagnostic only.
    Returns True if output is usable.
    """
    if not job_log or not lines:
        return True

    n        = len(lines)
    durs     = [max(0.0, l.get('end', 0) - l.get('start', 0)) for l in lines]
    min_d    = min(durs);  max_d = max(durs)
    ratio    = max_d / min_d if min_d > 0 else float('inf')
    n_short  = sum(1 for d in durs if d < _SAFEGUARD_MIN_DUR_S)
    n_nonmono = sum(
        1 for i in range(1, n)
        if lines[i].get('start', 0) < lines[i-1].get('end', 0) - 0.01
    )
    n_big_gap = sum(
        1 for i in range(1, n)
        if lines[i].get('start', 0) - lines[i-1].get('end', 0) > 3.0
    )

    failures: list[str] = []
    if n_short:
        failures.append(f"{n_short} lines < {_SAFEGUARD_MIN_DUR_S}s")
    if isinstance(ratio, float) and ratio > 20:
        failures.append(f"max/min ratio={ratio:.0f}x")
    if n_nonmono:
        failures.append(f"{n_nonmono} non-monotonic boundaries")
    if n_big_gap > 2:
        failures.append(f"{n_big_gap} gaps > 3s")

    usable = not failures
    if failures:
        job_log.warning(
            f"[sanity] FAIL ({n} lines): {' | '.join(failures)} — karaoke timing may be unusable",
            stage="align",
        )
    else:
        job_log.info(
            f"[sanity] PASS: {n} lines  ratio={ratio:.1f}x  no anomalies",
            stage="align",
        )
    return usable


# ── Boundary audit ────────────────────────────────────────────────────────────

_AUDIT_SHORT_MS   = 500    # flag lines shorter than this (ms)
_AUDIT_GAP_S      = 2.0    # flag inter-line gaps larger than this (s)
_AUDIT_DRIFT_TOL  = 0.01   # non-monotonic tolerance — ignore rounding noise (s)


def _audit_boundaries(lines: list[dict], job_log) -> None:
    """
    Emit a compact per-line timing table and anomaly report after alignment.

    Runs BEFORE gap-bridging so the raw acoustic boundaries are visible.
    Each line is printed with:
        L## start–end dur delta_from_prev src  text  [flags]

    Flags:
        ⚑NEGDUR   duration ≤ 0
        ⚑SHORT    duration < _AUDIT_SHORT_MS ms
        ⚑NONMONO  start < previous end (non-monotonic — root cause of drift)
        ⚑OVERLAP  start < previous end by > tolerance (alias for severity)
        ⚑GAP      gap to previous line > _AUDIT_GAP_S s

    The FIRST non-monotonic line is marked ← FIRST_DRIFT.
    """
    if not job_log or not lines:
        return

    n = len(lines)
    job_log.info(
        f"[AUDIT] ══════ boundary audit: {n} lines ══════",
        stage="align",
    )

    first_drift_idx: int      = -1
    anomaly_lines:   list[int] = []
    prev_end: float            = lines[0].get('start', 0.0)

    for idx, line in enumerate(lines):
        start = line.get('start', 0.0)
        end   = line.get('end',   0.0)
        dur   = end - start
        delta = start - prev_end          # positive = gap, negative = overlap
        src   = line.get('_timing_source', '?')[:14]
        text  = line.get('text', '')[:30]

        flags: list[str] = []

        if dur <= 0:
            flags.append('⚑NEGDUR')
        elif dur < _AUDIT_SHORT_MS / 1000:
            flags.append('⚑SHORT')

        if idx > 0:
            if delta < -_AUDIT_DRIFT_TOL:
                flags.append('⚑NONMONO')
                if first_drift_idx < 0:
                    first_drift_idx = idx
            if delta > _AUDIT_GAP_S:
                flags.append('⚑GAP')

        if flags:
            anomaly_lines.append(idx)

        drift_marker = '  ← FIRST_DRIFT' if idx == first_drift_idx else ''
        flag_str     = ('  ' + ' '.join(flags)) if flags else ''
        delta_str    = f"{delta * 1000:+.0f}ms" if idx > 0 else '    ---'

        level = 'warning' if flags else 'info'
        getattr(job_log, level)(
            f"[AUDIT] L{idx:02d}  {start:.3f}–{end:.3f}s  "
            f"dur={dur:.3f}s  prev_end={prev_end:.3f}s  Δ={delta_str:>8}  "
            f"src={src:<14}  '{text}'"
            f"{flag_str}{drift_marker}",
            stage="align",
        )

        prev_end = end

    # Summary
    if anomaly_lines:
        job_log.warning(
            f"[AUDIT] {len(anomaly_lines)} anomalous line(s) at indices "
            f"{anomaly_lines}"
            + (f" — FIRST_DRIFT at L{first_drift_idx:02d}" if first_drift_idx >= 0 else ""),
            stage="align",
        )
    else:
        job_log.info(
            f"[AUDIT] no anomalies — all {n} boundaries monotonic and well-formed",
            stage="align",
        )

    job_log.info(
        f"[AUDIT] ══════ end audit ══════",
        stage="align",
    )


# ── Public entry point ─────────────────────────────────────────────────────────

def force_align(
    segments:        AlignedSegments,
    audio_path:      Path,
    official_lyrics: Optional[list[dict]] = None,
    progress_cb:     Optional[Callable[[int], None]] = None,
    job_log=None,
) -> tuple[AlignedSegments, AlignmentMethod]:
    """
    Attach word-level timestamps to segments using the best available aligner.
    If official_lyrics are provided, uses them for acoustic forced alignment
    instead of Whisper's garbled text, giving much more precise timestamps.
    Returns (enriched_segments, method_name).
    """
    # Official lyrics path: replace Whisper text with correct lyrics and re-align
    if official_lyrics:
        try:
            result, method = _align_whisperx_with_official_lyrics(
                segments, official_lyrics, audio_path, job_log
            )
            if progress_cb:
                progress_cb(100)
            return result, method
        except Exception as exc:
            if job_log:
                job_log.warning(
                    f"[align] Official lyrics alignment failed, falling back to standard: {exc}",
                    stage="transcribe",
                )

    # Standard path: align Whisper's own transcription
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
