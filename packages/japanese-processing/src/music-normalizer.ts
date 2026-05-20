/**
 * Music-lyrics-specific normalization.
 *
 * Song lyrics differ from standard Japanese text in many ways:
 *   - Prolonged sounds written stylistically (あ〜〜〜、ラーーー)
 *   - Repeated characters for emphasis (笑笑笑、ねぇねぇ)
 *   - Missing or non-standard punctuation
 *   - Archaic / poetic kanji (儚い, 彷徨う)
 *   - Slang and dialect forms (ヤバい, チョー)
 *   - Partial sentences / fragments
 *   - Mixed scripts in unusual ways (Romaji mid-sentence)
 *   - Special notation (〈サビ〉, ※繰り返し)
 *
 * This normalizer produces two outputs:
 *   normalized  — cleaned text used for NLP analysis
 *   original    — original kept for display
 *
 * The goal is to make kuromoji/wanakana work better, NOT to alter the lyrics
 * for display purposes.
 */

// ── Regex patterns ─────────────────────────────────────────────────────────────

/** Section markers to strip before analysis */
const SECTION_MARKER_RE = /[〈《【\[（(][^〉》】\]）)]{1,20}[〉》】\]）)]/g;

/** Repeat notation (※ marks, 繰り返し, etc.) */
const REPEAT_MARKER_RE  = /[※×]\s*(?:繰り返し|repeat)?|＊+/gi;

/** Three or more of the same character → two */
const TRIPLE_REPEAT_RE  = /(.)\1{2,}/g;

/** Prolonged marks not followed by more kana (trailing decoration) */
const TRAILING_PROLONGED_RE = /[〜ー～]+$/g;

/** Excessive punctuation runs */
const PUNCT_RUN_RE = /[。、！？!?…]{3,}/g;

/** Musical notation embedded in lyrics */
const MUSICAL_NOTE_RE = /[♩♪♫♬♭♮♯]/g;

/** Common section labels that appear inline */
const INLINE_LABEL_RE  =
  /^(?:サビ|Aメロ|Bメロ|Cメロ|イントロ|アウトロ|ブリッジ|chorus|verse|bridge|intro|outro)\s*[:：]?\s*/i;

// ── Slang and dialect normalization ───────────────────────────────────────────

/**
 * Normalize common slang / informal forms so kuromoji can parse them.
 * Keep original for display; only use normalized for analysis.
 */
const SLANG_MAP: Array<[RegExp, string]> = [
  [/じゃん/g,   "じゃない"],   // assertion / negative
  [/だろ/g,     "だろう"],
  [/ちゃ/g,     "てしまう"],
  [/てんの/g,   "ているの"],
  [/んだ/g,     "のだ"],
  [/てる/g,     "ている"],
  [/でる/g,     "でいる"],
  [/ちゃう/g,   "てしまう"],
  [/でも(?=さ|ね|よ)/g, "でも"], // colloquial emphasis particles — keep as-is
  [/ぜ$/g,      "ぞ"],         // masculine sentence-final particle variant
];

// ── Analysis helpers ──────────────────────────────────────────────────────────

/** Detect poetic / archaic kanji (often appear in song lyrics). */
const ARCHAIC_KANJI = /[儚旋律奏魂漣煌綺纏彷徨彼岸悠翔曙暁刹那逢瀬逍遙]/;
export function hasArchaicKanji(text: string): boolean {
  return ARCHAIC_KANJI.test(text);
}

/** Detect slang patterns in the original text. */
const SLANG_PATTERNS = /じゃん|ヤバい?|チョー|めっちゃ|超|ガチ|マジ|ウザ|キモ|ダサ/i;
export function hasSlang(text: string): boolean {
  return SLANG_PATTERNS.test(text);
}

/** Detect unusual kanji (outside common-use Jōyō list). */
const JOYO_RE = /[一-鿿]/;
const UNUSUAL_KANJI_RE = /[㐀-䶿]/u;  // CJK Extension A (rare kanji)
export function hasUnusualKanji(text: string): boolean {
  return UNUSUAL_KANJI_RE.test(text);
}

// ── Main normalizer ────────────────────────────────────────────────────────────

export interface NormalizeResult {
  normalized:       string;
  original:         string;
  hasSlang:         boolean;
  hasArchaicKanji:  boolean;
  hasUnusualKanji:  boolean;
  wasModified:      boolean;
}

export function normalizeMusicLyric(text: string): NormalizeResult {
  const original = text;
  let s = text;

  // 1. Section markers / repeat markers
  s = s.replace(SECTION_MARKER_RE, "");
  s = s.replace(REPEAT_MARKER_RE,  "");
  s = s.replace(INLINE_LABEL_RE,   "");
  s = s.replace(MUSICAL_NOTE_RE,   "");

  // 2. Whitespace normalisation (preserve line breaks)
  s = s.replace(/[ \t　]+/g, " ").trim();

  // 3. Punctuation noise
  s = s.replace(PUNCT_RUN_RE,           "。");
  s = s.replace(TRAILING_PROLONGED_RE,  "");

  // 4. Repeated character collapse (3+ → 2 for analysis; still implies emphasis)
  s = s.replace(TRIPLE_REPEAT_RE, "$1$1");

  // 5. Slang normalization (only for analysis; original kept)
  // We do NOT apply SLANG_MAP to the normalized form used for display
  // but we DO apply it for the string we pass to kuromoji.
  // This is handled in the analyzer by passing normalized to tokenizer.

  return {
    normalized:      s.trim(),
    original,
    hasSlang:        hasSlang(original),
    hasArchaicKanji: hasArchaicKanji(original),
    hasUnusualKanji: hasUnusualKanji(original),
    wasModified:     s.trim() !== original.trim(),
  };
}

/**
 * Prepare text specifically for kuromoji tokenization.
 * Applies slang map on top of normalizeMusicLyric.
 */
export function prepareForTokenization(normalized: string): string {
  let s = normalized;
  for (const [pattern, replacement] of SLANG_MAP) {
    s = s.replace(pattern, replacement);
  }
  return s;
}
