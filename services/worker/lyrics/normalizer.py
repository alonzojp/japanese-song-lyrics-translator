"""
Japanese lyrics text normalizer.

Handles:
- Unicode normalization (NFKC)
- Full-width / half-width character normalization
- Katakana ↔ Hiragana (optional)
- Punctuation cleanup
- Whitespace collapse
- Common lyrics noise removal
"""
from __future__ import annotations

import re
import unicodedata

try:
    import jaconv
    _HAS_JACONV = True
except ImportError:
    _HAS_JACONV = False


# ── Regex patterns ─────────────────────────────────────────────────────────────

# Annotations: [Chorus], [Verse 1], （サビ）, 【Bメロ】, etc.
_ANNOTATION_RE = re.compile(
    r"[\[【「『（(][^】\]」』）)]*[\]】」』）)]",
    re.UNICODE,
)

# Repeated dashes/tildes used as section separators
_SEPARATOR_RE = re.compile(r"^[-─━=～〜＝]{3,}\s*$")

# Leading/trailing whitespace per line
_EDGE_SPACE_RE = re.compile(r"^[\s　]+|[\s　]+$", re.MULTILINE)

# Multiple consecutive blank lines → single blank line
_MULTI_BLANK_RE = re.compile(r"\n{3,}")

# Multiple spaces within a line (includes ideographic space)
_MULTI_SPACE_RE = re.compile(r"[ \t　]{2,}")

# Time codes sometimes left in lyrics: [00:30] or (1:30)
_TIMECODE_RE = re.compile(r"[\[(（]\d{1,2}:\d{2}(?:\.\d+)?[\])）]")

# ── Character maps ─────────────────────────────────────────────────────────────

# Normalize variant dashes / apostrophes
_CHAR_MAP: dict[str, str] = {
    "–": "—",   # en-dash → em-dash
    "…": "…",   # ellipsis
    "・": "・",  # katakana middle dot variants
    "･": "・",
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
}
_CHAR_TABLE = str.maketrans(_CHAR_MAP)


# ── Public API ─────────────────────────────────────────────────────────────────

def normalize_text(text: str, *, kata_to_hira: bool = False) -> str:
    """
    Full normalization pipeline for a lyrics block.

    Steps (in order):
      1. NFKC Unicode normalization
      2. Char-map substitutions (dashes, quotes)
      3. Time-code removal
      4. Annotation removal
      5. Separator line removal
      6. Whitespace collapse
      7. Optional: katakana → hiragana (for matching / comparison)
    """
    # 1. NFKC (collapses full-width ASCII, compatibility chars, etc.)
    text = unicodedata.normalize("NFKC", text)

    # 2. Character substitutions
    text = text.translate(_CHAR_TABLE)

    # 3. Remove inline time codes
    text = _TIMECODE_RE.sub("", text)

    # 4. Strip annotation brackets
    text = _ANNOTATION_RE.sub("", text)

    # 5. Remove separator lines
    lines = text.splitlines()
    lines = [ln for ln in lines if not _SEPARATOR_RE.match(ln)]
    text = "\n".join(lines)

    # 6. Edge whitespace + multi-space collapse
    text = _EDGE_SPACE_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    text = text.strip()

    # 7. Optional kana conversion
    if kata_to_hira and _HAS_JACONV:
        text = jaconv.kata2hira(text)

    return text


def normalize_line(line: str, *, kata_to_hira: bool = False) -> str:
    """Normalize a single lyric line (no multi-line processing)."""
    line = unicodedata.normalize("NFKC", line)
    line = line.translate(_CHAR_TABLE)
    line = _TIMECODE_RE.sub("", line)
    line = _ANNOTATION_RE.sub("", line)
    line = _EDGE_SPACE_RE.sub("", line).strip()
    line = _MULTI_SPACE_RE.sub(" ", line)
    if kata_to_hira and _HAS_JACONV:
        line = jaconv.kata2hira(line)
    return line


def is_noise_line(line: str) -> bool:
    """True if a line should be dropped entirely (empty, separator, ad text)."""
    stripped = line.strip()
    if not stripped:
        return True
    if _SEPARATOR_RE.match(stripped):
        return True
    # Pure annotation (nothing left after removing brackets)
    if not _ANNOTATION_RE.sub("", stripped).strip():
        return True
    # Very short non-Japanese lines are often noise ("by", "feat.", "prod.", etc.)
    if len(stripped) <= 3 and not _is_japanese(stripped):
        return True
    return False


def text_to_lines(text: str) -> list[str]:
    """Split normalized text into non-empty lines."""
    return [ln for ln in text.splitlines() if not is_noise_line(ln)]


def japanese_char_ratio(text: str) -> float:
    """Return fraction of characters that are CJK / kana (0.0–1.0)."""
    if not text:
        return 0.0
    ja = sum(1 for c in text if _is_japanese(c))
    return ja / len(text)


def _is_japanese(text: str) -> bool:
    for c in text:
        cp = ord(c)
        if (
            0x3040 <= cp <= 0x309F  # hiragana
            or 0x30A0 <= cp <= 0x30FF  # katakana
            or 0x4E00 <= cp <= 0x9FFF  # CJK unified
            or 0x3400 <= cp <= 0x4DBF  # CJK ext A
        ):
            return True
    return False
