/**
 * Text conversion utilities wrapping wanakana.
 *
 * wanakana is the canonical kana/romaji conversion library.
 * It's written in TypeScript, works in both Node.js and browser.
 *
 * This module re-exports the most useful functions and adds
 * Japanese-music-lyric-specific helpers.
 */
export {
  toRomaji,
  toHiragana,
  toKatakana,
  toKana,
  isHiragana,
  isKatakana,
  isKanji,
  isJapanese,
  isRomaji,
  stripOkurigana,
  tokenize as wanakanaTokenize,
} from "wanakana";

import { toRomaji as _toRomaji, toHiragana, isKanji as _isKanji } from "wanakana";

// ── Japanese character detection ───────────────────────────────────────────────

const KANJI_RE = /[一-鿿㐀-䶿]/;

export function containsKanji(text: string): boolean {
  return KANJI_RE.test(text);
}

/** True for particles, auxiliaries, and kana-only small words. */
export function isKanaOnly(text: string): boolean {
  return /^[ぁ-んァ-ヶー]+$/.test(text);
}

// ── Katakana to hiragana ───────────────────────────────────────────────────────

export function kataToHira(text: string): string {
  return text.replace(/[ァ-ヶ]/g, (c) => String.fromCharCode(c.charCodeAt(0) - 0x60));
}

// ── Hepburn romaji from reading ────────────────────────────────────────────────

/**
 * Convert a hiragana/katakana reading to Hepburn romaji.
 * Uses wanakana which handles all edge cases (っ, long vowels, etc.)
 */
export function readingToRomaji(reading: string): string {
  if (!reading) return "";
  const hira = kataToHira(reading);
  return _toRomaji(hira, {
    customRomajiMapping: {
      // Common music pronunciation
      じゃ: "ja", じゅ: "ju", じょ: "jo",
    },
  });
}

// ── Furigana alignment from kuromoji reading ───────────────────────────────────

/**
 * Given a surface form and its katakana reading (from kuromoji),
 * produce a `food(たべもの)` style annotation string that alignFurigana
 * in furigana.ts can then convert to <ruby> HTML.
 *
 * Returns null for kana-only surfaces (no annotation needed).
 */
export function buildFuriganaAnnotation(surface: string, katakanaReading: string): string | null {
  if (!surface || !katakanaReading) return null;
  if (!containsKanji(surface)) return null;

  const hiraReading = kataToHira(katakanaReading);

  // Try to align: strip matching kana suffix so annotation covers only kanji
  let si = surface.length - 1;
  let ri = hiraReading.length - 1;
  while (
    si > 0 &&
    ri > 0 &&
    surface[si] === hiraReading[ri] &&
    /[ぁ-んァ-ン]/.test(surface[si]!)
  ) {
    si--;
    ri--;
  }

  const kanjiPart   = surface.slice(0, si + 1);
  const kanaSuffix  = surface.slice(si + 1);
  const kanjiRead   = hiraReading.slice(0, ri + 1);

  if (!kanjiPart || !kanjiRead || kanjiRead === kanjiPart) {
    // Whole-word annotation
    return `${surface}(${hiraReading})`;
  }

  return `${kanjiPart}(${kanjiRead})${kanaSuffix}`;
}

// ── Line-level romaji ──────────────────────────────────────────────────────────

/**
 * Convert a full Japanese line to Hepburn romaji.
 * Handles mixed kana/kanji/Latin text.
 */
export function lineToRomaji(text: string): string {
  // wanakana's toRomaji handles mixed text gracefully
  return _toRomaji(kataToHira(text), {
    upcaseKatakana: false,
    convertLongVowelMark: true,
  });
}
