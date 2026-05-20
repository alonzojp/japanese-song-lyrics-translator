/**
 * kuromoji-based morphological analyzer.
 *
 * kuromoji (0.1.x) is CJS — this module handles ESM interop cleanly.
 * Falls back to Intl.Segmenter when kuromoji is unavailable (browser / no dict).
 */
import type { PartOfSpeech } from "./models.js";

// ── kuromoji types ─────────────────────────────────────────────────────────────

interface KuroToken {
  word_id:         number;
  word_type:       string;
  word_position:   number;
  surface_form:    string;
  pos:             string;
  pos_detail_1:    string;
  pos_detail_2:    string;
  pos_detail_3:    string;
  conjugated_type: string;
  conjugated_form: string;
  basic_form:      string;
  reading:         string;
  pronunciation:   string;
}

interface KuroTokenizer { tokenize(text: string): KuroToken[]; }
interface KuroBuilder   { build(cb: (err: Error | null, t: KuroTokenizer) => void): void; }
interface KuromojiMod   { builder(opts: { dicPath: string }): KuroBuilder; }

// ── Singleton ──────────────────────────────────────────────────────────────────

let _tokenizer: KuroTokenizer | null = null;
let _initPromise: Promise<KuroTokenizer | null> | null = null;

function dicPath(): string {
  const cwd = (typeof process !== "undefined" && process.cwd) ? process.cwd() : ".";
  return `${cwd}/node_modules/kuromoji/dict`;
}

async function buildTokenizer(): Promise<KuroTokenizer | null> {
  try {
    const mod  = await import("kuromoji") as { default?: KuromojiMod } & KuromojiMod;
    const kuro = (mod.default ?? mod) as KuromojiMod;
    return new Promise<KuroTokenizer | null>((resolve) => {
      kuro.builder({ dicPath: dicPath() }).build((err, t) => resolve(err ? null : t));
    });
  } catch {
    return null;
  }
}

export async function getTokenizer(): Promise<KuroTokenizer | null> {
  if (_tokenizer) return _tokenizer;
  if (!_initPromise) _initPromise = buildTokenizer().then((t) => { _tokenizer = t; return t; });
  return _initPromise;
}

// ── POS mapping ────────────────────────────────────────────────────────────────

const POS_MAP: Record<string, PartOfSpeech> = {
  "名詞":     "noun",     "動詞":     "verb",          "形容詞":   "adjective",
  "形容動詞": "na-adjective", "副詞": "adverb",        "助詞":     "particle",
  "助動詞":   "auxiliary","接続詞":   "conjunction",    "感動詞":   "interjection",
  "接頭詞":   "prefix",   "接尾詞":   "suffix",        "記号":     "symbol",
  "フィラー": "filler",
};

export function mapPos(posJa: string): PartOfSpeech {
  return POS_MAP[posJa] ?? "other";
}

export function isFunctionalPos(posJa: string, detail1: string): boolean {
  return (
    posJa === "助詞" || posJa === "助動詞" ||
    (posJa === "動詞" && detail1 === "非自立") ||
    (posJa === "名詞" && detail1 === "非自立")
  );
}

// ── Raw token ──────────────────────────────────────────────────────────────────

export interface RawToken {
  surface:        string;
  reading:        string;
  pronunciation:  string;
  baseForm:       string;
  posJa:          string;
  posDetail1:     string;
  posDetail2:     string;
  posDetail3:     string;
  conjugatedType: string;
  conjugatedForm: string;
  isUnknown:      boolean;
}

// ── Public entry point ─────────────────────────────────────────────────────────

export async function tokenizeRaw(text: string): Promise<RawToken[]> {
  const tokenizer = await getTokenizer();
  if (!tokenizer) return _fallback(text);

  return tokenizer.tokenize(text).map((t) => ({
    surface:        t.surface_form,
    reading:        t.reading         ?? "",
    pronunciation:  t.pronunciation   ?? t.reading ?? "",
    baseForm:       t.basic_form      ?? t.surface_form,
    posJa:          t.pos             ?? "名詞",
    posDetail1:     t.pos_detail_1    ?? "",
    posDetail2:     t.pos_detail_2    ?? "",
    posDetail3:     t.pos_detail_3    ?? "",
    conjugatedType: t.conjugated_type ?? "",
    conjugatedForm: t.conjugated_form ?? "",
    isUnknown:      t.word_type === "UNKNOWN",
  }));
}

function _fallback(text: string): RawToken[] {
  if (typeof Intl !== "undefined" && "Segmenter" in Intl) {
    const seg = new Intl.Segmenter("ja", { granularity: "word" });
    return [...seg.segment(text)].filter((s) => s.isWordLike).map((s) => ({
      surface: s.segment, reading: "", pronunciation: "", baseForm: s.segment,
      posJa: "名詞", posDetail1: "", posDetail2: "", posDetail3: "",
      conjugatedType: "", conjugatedForm: "", isUnknown: true,
    }));
  }
  return [...text].map((c) => ({
    surface: c, reading: "", pronunciation: "", baseForm: c,
    posJa: "記号", posDetail1: "", posDetail2: "", posDetail3: "",
    conjugatedType: "", conjugatedForm: "", isUnknown: true,
  }));
}
