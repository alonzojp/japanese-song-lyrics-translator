/**
 * Japanese pitch accent support.
 *
 * Pitch accent in Japanese determines which morae are high vs low.
 * Full NHK accent data is ~8 MB and proprietary; this module uses:
 *   1. A small inline table for high-frequency words (~300 entries)
 *   2. Rule-based heuristics for verbs, adjectives, and loanwords
 *   3. Returns null for unknown words rather than guessing wrong
 *
 * Pattern encoding (NHK-style drop position):
 *   0 = 平板型 (heiban)    — rises on mora 2, stays high
 *   1 = 頭高型 (atamadaka) — high on mora 1, drops on mora 2
 *   2 = 中高型 (nakadaka)  — rises on mora 2, drops on mora 3
 *   3 = 尾高型 (odaka)     — rises on mora 2, stays high, drops after particle
 *   (2+ in general: drops after mora N)
 */
import type { PitchAccent } from "./models.js";
import { kataToHira } from "./converter.js";

// ── Inline pitch accent table (drop position keyed by hiragana base form) ──────
const PITCH_TABLE: Record<string, number> = {
  // N5 words
  "わたし":3,"ぼく":2,"おれ":2,"きみ":2,"かれ":0,"かのじょ":0,
  "にほん":2,"がっこう":3,"うち":0,"いえ":0,"へや":0,"まち":2,
  "はな":1,"き":1,"やま":2,"うみ":1,"そら":1,"つき":0,
  "こころ":0,"ゆめ":2,"ひかり":0,"かぜ":1,"みち":0,
  "てんき":1,"あめ":2,"ゆき":0,"はる":0,"なつ":2,"あき":1,"ふゆ":0,
  "おとこ":0,"おんな":0,"こ":1,"ひと":0,
  "いく":0,"くる":1,"かえる":0,"みる":1,"きく":2,
  "よむ":0,"かく":0,"はなす":0,"おもう":0,"しる":0,
  "ある":0,"いる":0,"する":0,"なる":0,
  // Common lyric words
  "あい":1,"こい":1,"なみだ":0,"うた":0,"こえ":1,"ひびく":0,
  "かがやく":0,"ながれる":3,"もえる":0,"ちる":1,"さく":1,
  "いのち":1,"ちから":3,"きもち":0,
  "えいえん":0,"ちかい":0,"まもる":0,"たたかう":0,
  "さびしい":0,"かなしい":0,"うれしい":0,"たのしい":3,
  "やさしい":4,"つよい":2,"よわい":2,
};

const LABEL_MAP: Record<number, PitchAccent["label"]> = {
  0: "平板型",
  1: "頭高型",
};

function toLabel(drop: number, moraLen: number): PitchAccent["label"] {
  if (drop === 0) return "平板型";
  if (drop === 1) return "頭高型";
  if (drop === moraLen) return "尾高型";
  return "中高型";
}

// ── Mora counting ──────────────────────────────────────────────────────────────

// Small kana that attach to the previous mora
const SMALL_KANA = new Set([..."ぁぃぅぇぉゃゅょっァィゥェォャュョッ"]);

function countMora(hira: string): number {
  let count = 0;
  for (const c of hira) {
    if (!SMALL_KANA.has(c)) count++;
  }
  return count;
}

// ── Verb/adjective patterns ────────────────────────────────────────────────────
// Most 2-mora verbs: 起きる (2), 見る (1), etc.
// Heuristics rather than lookup:
function verbHeuristic(hira: string, posDetail: string): number | null {
  if (!hira) return null;
  const morae = countMora(hira);

  // Suru-verbs (〜する): usually 0
  if (hira.endsWith("する")) return 0;
  // Kuru (くる): 1
  if (hira === "くる") return 1;
  // Most godan verbs: 0
  if (/[うくすつぬぶむ]$/.test(hira) && morae <= 3) return 0;
  // Most ichidan (〜る): depends on length
  if (hira.endsWith("る")) return morae <= 2 ? 1 : 0;

  return null;
}

function iAdjHeuristic(hira: string): number | null {
  if (!hira.endsWith("い")) return null;
  const morae = countMora(hira);
  // 3-mora i-adj (like 高い たかい): 2
  if (morae === 3) return 2;
  // 4-mora (like 難しい): usually 4
  if (morae === 4) return 4;
  return null;
}

// ── Public API ─────────────────────────────────────────────────────────────────

/**
 * Look up or estimate pitch accent for a token.
 * Returns null when the pattern is genuinely unknown.
 */
export function lookupPitchAccent(
  baseForm:  string,
  posJa:     string,
  posDetail: string = "",
): PitchAccent | null {
  const hira = kataToHira(baseForm.toLowerCase());
  const morae = countMora(hira);
  if (morae === 0) return null;

  // 1. Exact table lookup
  const drop = PITCH_TABLE[hira];
  if (drop !== undefined) {
    return { dropPosition: drop, label: toLabel(drop, morae), estimated: false };
  }

  // 2. Verb/adjective heuristics
  let estimated: number | null = null;
  if (posJa === "動詞") {
    estimated = verbHeuristic(hira, posDetail);
  } else if (posJa === "形容詞") {
    estimated = iAdjHeuristic(hira);
  } else if (posJa === "名詞" && /^[ァ-ヶー]+$/.test(baseForm)) {
    // Loanwords (katakana nouns): tend to be flat or last-syllable high
    estimated = morae <= 3 ? 0 : morae - 1;
  }

  if (estimated !== null) {
    return { dropPosition: estimated, label: toLabel(estimated, morae), estimated: true };
  }

  return null;
}
