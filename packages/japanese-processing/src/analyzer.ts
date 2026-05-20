/**
 * Main NLP pipeline — assembles all modules into analyzeLine() and analyzeLines().
 *
 * For each lyric line it produces:
 *   - kuromoji tokenization
 *   - per-token furigana, reading, romaji
 *   - dictionary meaning + JLPT level (via Jotoba, cached)
 *   - pitch accent (inline table + heuristics)
 *   - line-level furigana HTML + romaji string
 *
 * Music lyric mode:
 *   - Tolerates fragments, slang, unusual kanji
 *   - Normalizes before tokenization, preserves original for display
 *   - Aggressive caching: identical lines are never analyzed twice
 */
import { tokenizeRaw, mapPos, isFunctionalPos } from "./tokenizer.js";
import { buildFuriganaAnnotation, kataToHira, readingToRomaji, lineToRomaji, containsKanji } from "./converter.js";
import { furiganaToRuby, buildRubyHTML } from "./furigana.js";
import { lookupJlpt, mostAdvancedJlpt } from "./jlpt.js";
import { lookupPitchAccent } from "./pitch-accent.js";
import {
  getDefaultDictionaryService,
  type DictionaryService,
} from "./dictionary.js";
import { normalizeMusicLyric, prepareForTokenization } from "./music-normalizer.js";
import { lineCache } from "./cache.js";
import type { AnalyzedLine, JapaneseToken, JlptLevel } from "./models.js";

// ── Options ────────────────────────────────────────────────────────────────────

export interface AnalyzerOptions {
  /** Skip dictionary API calls (faster, no JLPT from API) */
  skipDictionary?:  boolean;
  /** Skip pitch accent lookup */
  skipPitchAccent?: boolean;
  /** Custom dictionary service */
  dictionary?:      DictionaryService;
  /** Mark as music lyric (enables music normalization) */
  musicMode?:       boolean;
}

const DEFAULT_OPTIONS: Required<AnalyzerOptions> = {
  skipDictionary:  false,
  skipPitchAccent: false,
  dictionary:      null as unknown as DictionaryService,  // filled at call time
  musicMode:       true,
};

// ── Single-line analysis ───────────────────────────────────────────────────────

/**
 * Fully analyze one Japanese lyric line.
 *
 * @param text     Original lyric line text
 * @param options  Pipeline options
 */
export async function analyzeLine(
  text:    string,
  options: AnalyzerOptions = {},
): Promise<AnalyzedLine> {
  const opts: Required<AnalyzerOptions> = {
    ...DEFAULT_OPTIONS,
    ...options,
    dictionary: options.dictionary ?? getDefaultDictionaryService(),
  };

  // ── Cache lookup ─────────────────────────────────────────────────────────────
  const cacheKey = `${opts.skipDictionary}:${text}`;
  const cached   = lineCache.get(cacheKey);
  if (cached) return cached as AnalyzedLine;

  // ── Music normalization ───────────────────────────────────────────────────────
  const { normalized, hasSlang, hasArchaicKanji, hasUnusualKanji } =
    opts.musicMode
      ? normalizeMusicLyric(text)
      : { normalized: text, hasSlang: false, hasArchaicKanji: false, hasUnusualKanji: false };

  const forTokenization = opts.musicMode ? prepareForTokenization(normalized) : normalized;

  // ── Tokenization ──────────────────────────────────────────────────────────────
  const rawTokens = await tokenizeRaw(forTokenization);

  // ── Dictionary batch lookup ───────────────────────────────────────────────────
  let dictMap = new Map<string, { meanings: string[]; jlpt: JlptLevel | null }>();
  if (!opts.skipDictionary) {
    const lookupTargets = rawTokens
      .filter((t) => !isFunctionalPos(t.posJa, t.posDetail1) && t.baseForm !== "*")
      .map((t) => t.baseForm !== "*" && t.baseForm ? t.baseForm : t.surface);

    const entries = await opts.dictionary.lookupBatch([...new Set(lookupTargets)]);
    for (const [word, entry] of entries) {
      if (entry) {
        dictMap.set(word, { meanings: entry.meanings, jlpt: entry.jlpt });
      }
    }
  }

  // ── Build JapaneseToken array ─────────────────────────────────────────────────
  const tokens: JapaneseToken[] = rawTokens.map((raw) => {
    const hiraReading = raw.reading ? kataToHira(raw.reading) : raw.surface;
    const hiraBase    = raw.baseForm && raw.baseForm !== "*"
                          ? kataToHira(raw.baseForm) : hiraReading;
    const romaji      = readingToRomaji(hiraReading);
    const furiganaAnnot = containsKanji(raw.surface)
                            ? buildFuriganaAnnotation(raw.surface, raw.reading || raw.surface)
                            : null;
    const partOfSpeech  = mapPos(raw.posJa);
    const isFunctional  = isFunctionalPos(raw.posJa, raw.posDetail1);

    // JLPT: dict API first, then inline table
    const lookupKey   = raw.baseForm && raw.baseForm !== "*" ? raw.baseForm : raw.surface;
    const dictEntry   = dictMap.get(lookupKey);
    const jlptFromApi = dictEntry?.jlpt ?? null;
    const jlptInline  = lookupJlpt(raw.surface, hiraReading, raw.baseForm);
    const jlpt        = jlptFromApi ?? jlptInline;

    const meanings    = dictEntry?.meanings ?? [];
    const meaning     = meanings[0] ?? null;

    const pitchAccent = opts.skipPitchAccent
      ? null
      : lookupPitchAccent(raw.baseForm || raw.surface, raw.posJa, raw.posDetail1);

    // Confidence: lower for unknown words, very short tokens, or kana-only
    const confidence  = raw.isUnknown ? 0.4
                      : isFunctional  ? 0.95
                      : raw.reading   ? 0.85
                      : 0.55;

    return {
      surface:      raw.surface,
      reading:      hiraReading,
      furigana:     furiganaAnnot,
      romaji,
      baseForm:     raw.baseForm !== "*" ? (raw.baseForm ?? raw.surface) : raw.surface,
      partOfSpeech,
      posJa:        raw.posJa,
      posDetail:    raw.posDetail1,
      meaning,
      meanings,
      jlpt,
      isKanji:      containsKanji(raw.surface),
      isFunctional,
      pitchAccent,
      confidence,
    };
  });

  // ── Line-level furigana HTML ───────────────────────────────────────────────────
  const furiganaHtml = tokens
    .map((t) => {
      if (!t.furigana) return t.surface;
      return furiganaToRuby(t.furigana) ?? t.surface;
    })
    .join("");

  // ── Line-level romaji ──────────────────────────────────────────────────────────
  const romaji = lineToRomaji(normalized);

  // ── Overall JLPT level ────────────────────────────────────────────────────────
  const jlptLevel = mostAdvancedJlpt(tokens.map((t) => t.jlpt));

  const result: AnalyzedLine = {
    text:         text,
    normalized,
    tokens,
    furigana:     furiganaHtml,
    romaji,
    jlptLevel,
    isMusicLyric: opts.musicMode,
    hasSlang,
    hasUnusualKanji: hasUnusualKanji || hasArchaicKanji,
  };

  lineCache.set(cacheKey, result);
  return result;
}

// ── Batch analysis ─────────────────────────────────────────────────────────────

const MAX_CONCURRENT = 4;

/**
 * Analyze multiple lyric lines with bounded concurrency.
 * Returns results in the same order as input.
 */
export async function analyzeLines(
  texts:   string[],
  options: AnalyzerOptions = {},
): Promise<AnalyzedLine[]> {
  const results: AnalyzedLine[] = new Array(texts.length);

  for (let i = 0; i < texts.length; i += MAX_CONCURRENT) {
    const batch = texts.slice(i, i + MAX_CONCURRENT);
    const settled = await Promise.allSettled(
      batch.map((t) => analyzeLine(t, options))
    );
    for (let j = 0; j < settled.length; j++) {
      const r = settled[j]!;
      results[i + j] = r.status === "fulfilled"
        ? r.value
        : {
            text:            batch[j]!,
            normalized:      batch[j]!,
            tokens:          [],
            furigana:        batch[j]!,
            romaji:          "",
            jlptLevel:       null,
            isMusicLyric:    true,
            hasSlang:        false,
            hasUnusualKanji: false,
          };
    }
  }

  return results;
}

// ── Pipeline class ─────────────────────────────────────────────────────────────

/**
 * Reusable NLP pipeline with persistent options and dictionary service.
 * Ideal when processing many lines from the same song.
 */
export class NLPPipeline {
  private readonly opts: Required<AnalyzerOptions>;

  constructor(options: AnalyzerOptions = {}) {
    this.opts = {
      ...DEFAULT_OPTIONS,
      ...options,
      dictionary: options.dictionary ?? getDefaultDictionaryService(),
    };
  }

  analyzeLine(text: string): Promise<AnalyzedLine> {
    return analyzeLine(text, this.opts);
  }

  analyzeLines(texts: string[]): Promise<AnalyzedLine[]> {
    return analyzeLines(texts, this.opts);
  }

  /** Clear the analysis cache (useful after dictionary updates). */
  clearCache(): void {
    lineCache.clear();
  }
}
