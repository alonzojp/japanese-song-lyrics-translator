/**
 * Token and analysis types for Japanese NLP pipeline.
 *
 * Designed to match the product spec example:
 *   { surface, reading, furigana, meaning, jlpt }
 * Plus extended fields for richer downstream use.
 */

export type JlptLevel = "N1" | "N2" | "N3" | "N4" | "N5";

export type PartOfSpeech =
  | "noun"
  | "verb"
  | "adjective"         // i-adjective (形容詞)
  | "na-adjective"      // na-adjective (形容動詞)
  | "adverb"
  | "particle"
  | "auxiliary"
  | "conjunction"
  | "interjection"
  | "prefix"
  | "suffix"
  | "symbol"
  | "filler"
  | "other";

/** NHK-style pitch accent pattern. */
export interface PitchAccent {
  /** Mora index where pitch drops (0 = flat/heiban, 1 = atamadaka, 2+ = nakadaka/odaka). */
  dropPosition: number;
  /** Human-readable pattern label. */
  label: "平板型" | "頭高型" | "中高型" | "尾高型" | string;
  /** Whether this pattern is certain (from dictionary) or estimated. */
  estimated: boolean;
}

/** One morpheme produced by the analysis pipeline. */
export interface JapaneseToken {
  // ── Core (matches product spec) ─────────────────────────────────────────────
  surface:      string;           // word as it appears in the text
  reading:      string;           // hiragana reading
  furigana:     string | null;    // reading annotation for kanji (null for kana-only)
  romaji:       string;           // Hepburn romanization
  meaning:      string | null;    // primary English definition
  jlpt:         JlptLevel | null;

  // ── Extended ─────────────────────────────────────────────────────────────────
  baseForm:     string;           // dictionary / citation form
  partOfSpeech: PartOfSpeech;
  posJa:        string;           // raw Japanese POS label from kuromoji
  posDetail:    string;           // sub-category (固有名詞, 自立, etc.)
  meanings:     string[];         // all English definitions
  pitchAccent:  PitchAccent | null;
  isKanji:      boolean;          // contains at least one kanji character
  isFunctional: boolean;          // particle / auxiliary / copula (low semantic weight)
  confidence:   number;           // 0.0–1.0 how confident the reading is
}

/** Full analysis of one lyric line. */
export interface AnalyzedLine {
  text:         string;           // original text
  normalized:   string;           // music-normalized text used for analysis
  tokens:       JapaneseToken[];
  furigana:     string;           // full line with <ruby> HTML
  romaji:       string;           // full line romanized
  jlptLevel:    JlptLevel | null; // most advanced JLPT level present
  isMusicLyric: boolean;
  hasSlang:     boolean;
  hasUnusualKanji: boolean;
}

/** Compact token shape for serialization / storage. */
export interface TokenRecord {
  surface:  string;
  reading:  string;
  furigana: string | null;
  romaji:   string;
  meaning:  string | null;
  jlpt:     JlptLevel | null;
  pos:      PartOfSpeech;
  baseForm: string;
}

export function tokenToRecord(t: JapaneseToken): TokenRecord {
  return {
    surface:  t.surface,
    reading:  t.reading,
    furigana: t.furigana,
    romaji:   t.romaji,
    meaning:  t.meaning,
    jlpt:     t.jlpt,
    pos:      t.partOfSpeech,
    baseForm: t.baseForm,
  };
}
