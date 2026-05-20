// ── Models ─────────────────────────────────────────────────────────────────────
export type {
  JapaneseToken,
  AnalyzedLine,
  PitchAccent,
  PartOfSpeech,
  JlptLevel,
  TokenRecord,
} from "./models.js";
export { tokenToRecord } from "./models.js";

// ── Cache ──────────────────────────────────────────────────────────────────────
export { LRUCache, tokenCache, dictCache, lineCache } from "./cache.js";

// ── Tokenizer (kuromoji) ───────────────────────────────────────────────────────
export { tokenizeRaw, getTokenizer, mapPos, isFunctionalPos } from "./tokenizer.js";
export type { RawToken } from "./tokenizer.js";

// ── Converter (wanakana) ───────────────────────────────────────────────────────
export {
  toRomaji,
  toHiragana,
  toKatakana,
  isHiragana,
  isKatakana,
  isKanji,
  isJapanese,
  isRomaji,
  kataToHira,
  containsKanji,
  isKanaOnly,
  readingToRomaji,
  buildFuriganaAnnotation,
  lineToRomaji,
} from "./converter.js";

// ── Furigana ───────────────────────────────────────────────────────────────────
export {
  KANJI_RE,
  furiganaToRuby,
  alignFurigana,
  buildRubyHTML,
  buildRubyFromTokens,
  looksLikeReading,
} from "./furigana.js";

// ── JLPT ──────────────────────────────────────────────────────────────────────
export { lookupJlpt, mostAdvancedJlpt } from "./jlpt.js";

// ── Pitch accent ───────────────────────────────────────────────────────────────
export { lookupPitchAccent } from "./pitch-accent.js";

// ── Dictionary ────────────────────────────────────────────────────────────────
export {
  JotobaDictionaryService,
  getDefaultDictionaryService,
  setDictionaryService,
  primaryMeaning,
} from "./dictionary.js";
export type { DictionaryEntry, DictionaryService } from "./dictionary.js";

// ── Music normalizer ───────────────────────────────────────────────────────────
export {
  normalizeMusicLyric,
  prepareForTokenization,
  hasSlang,
  hasArchaicKanji,
  hasUnusualKanji,
} from "./music-normalizer.js";
export type { NormalizeResult } from "./music-normalizer.js";

// ── Segmenter (Intl.Segmenter fallback) ────────────────────────────────────────
export { segmentText } from "./segmenter.js";

// ── Main analyzer ─────────────────────────────────────────────────────────────
export { analyzeLine, analyzeLines, NLPPipeline } from "./analyzer.js";
export type { AnalyzerOptions } from "./analyzer.js";
