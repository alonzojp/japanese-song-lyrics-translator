import { toRomaji } from "@japanese-lyrics/japanese-processing";

// ── Character ranges ────────────────────────────────────────────────────────────
const KATA_START = 0x30A1;
const KATA_END   = 0x30F6;
const HIRA_OFFSET = 0x60;

// Patterns
const PUNCTUATION_RE  = /[。、！？!?.,…「」『』【】〈〉《》・\-\s　]+/g;
const PROLONGED_RE    = /ー+/g;
// Collapse 3+ consecutive identical characters (あああ → あ)
const REPEATED_RE     = /(.)\1{2,}/g;
const KANJI_RE        = /[一-鿿㐀-䶿]/;

// ── Core transforms ────────────────────────────────────────────────────────────

/** Convert katakana codepoints to hiragana equivalents. */
export function kataToHira(text: string): string {
  return text.replace(/[ァ-ヶ]/g, (c) =>
    String.fromCharCode(c.charCodeAt(0) - HIRA_OFFSET)
  );
}

/** Unicode NFKC normalization. Collapses full-width ASCII, compat chars. */
function nfkc(text: string): string {
  return text.normalize("NFKC");
}

/**
 * Full normalization pipeline for matching purposes.
 *
 * Steps:
 *   1. NFKC
 *   2. Katakana → hiragana
 *   3. Strip punctuation / whitespace
 *   4. Strip prolonged sound marks (ー) — 先生ー → 先生
 *   5. Collapse 3+ repeated characters
 *   6. Lowercase
 */
export function normalize(text: string): string {
  let s = nfkc(text);
  s = kataToHira(s);
  s = s.replace(PUNCTUATION_RE, "");
  s = s.replace(PROLONGED_RE, "");
  s = s.replace(REPEATED_RE, "$1$1"); // collapse to at most 2 repetitions
  s = s.toLowerCase();
  return s.trim();
}

/**
 * Light normalization — preserves word boundaries and punctuation.
 * Used for display, not matching.
 */
export function normalizeSoft(text: string): string {
  return nfkc(text).replace(/\s+/g, " ").trim();
}

/**
 * Convert text to romaji for fallback comparison.
 * Strips all non-ASCII after conversion.
 */
export function toNormalizedRomaji(text: string): string {
  const hira = kataToHira(nfkc(text));
  const rom  = toRomaji(hira);
  return rom.toLowerCase().replace(/[^a-z0-9]/g, "");
}

/** True if the string contains Japanese characters. */
export function hasJapanese(text: string): boolean {
  return /[぀-ゟ゠-ヿ一-鿿㐀-䶿]/.test(text);
}

/** Fraction of characters that are CJK or kana. */
export function japaneseRatio(text: string): number {
  if (!text.length) return 0;
  const count = [...text].filter((c) => {
    const cp = c.charCodeAt(0);
    return (
      (cp >= 0x3040 && cp <= 0x309F) ||
      (cp >= 0x30A0 && cp <= 0x30FF) ||
      (cp >= 0x4E00 && cp <= 0x9FFF) ||
      (cp >= 0x3400 && cp <= 0x4DBF)
    );
  }).length;
  return count / text.length;
}

/**
 * Normalize a sequence of strings in bulk.
 * Returns both the normalized strings and their romaji equivalents.
 */
export function normalizeAll(texts: string[]): { norm: string[]; romaji: string[] } {
  const norm   = texts.map(normalize);
  const romaji = texts.map(toNormalizedRomaji);
  return { norm, romaji };
}
