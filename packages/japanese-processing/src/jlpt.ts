/**
 * JLPT level lookup.
 *
 * Lookup order:
 *   1. Inline compact table (N5 complete + common N4/N3)
 *   2. Kanji-difficulty heuristic (counts N1/N2 Jōyō kanji)
 *   3. (Optional) Jotoba API call — used by the dictionary service layer
 *
 * The inline table uses hiragana base forms as keys for fast lookup.
 */
import type { JlptLevel } from "./models.js";
import { kataToHira } from "./converter.js";

// ── Compact JLPT vocabulary table ─────────────────────────────────────────────
// Format: "hiragana_base_form" → "N5"|"N4"|"N3"|"N2"|"N1"
// N5 and common N4 are included inline. N3–N1 use heuristics + API.
const JLPT_TABLE: Record<string, JlptLevel> = {
  // ── N5 ────────────────────────────────────────────────────────────────────
  "わたし":"N5","ぼく":"N5","おれ":"N5","きみ":"N5","あなた":"N5","かれ":"N5",
  "かのじょ":"N5","みんな":"N5","だれ":"N5","なに":"N5","なん":"N5",
  "どこ":"N5","いつ":"N5","なぜ":"N5","どう":"N5","どの":"N5",
  "いい":"N5","よい":"N5","わるい":"N5","おおきい":"N5","ちいさい":"N5",
  "あたらしい":"N5","ふるい":"N5","たかい":"N5","やすい":"N5","ながい":"N5",
  "みじかい":"N5","あつい":"N5","さむい":"N5","つよい":"N5","よわい":"N5",
  "はやい":"N5","おそい":"N5","むずかしい":"N5","かんたん":"N5","きれい":"N5",
  "すき":"N5","きらい":"N5","じょうず":"N5","へた":"N5","げんき":"N5",
  "ひと":"N5","おとこ":"N5","おんな":"N5","こ":"N5","おや":"N5",
  "ちち":"N5","はは":"N5","おとうさん":"N5","おかあさん":"N5","にほん":"N5",
  "がっこう":"N5","うち":"N5","いえ":"N5","へや":"N5","まち":"N5",
  "くに":"N5","きょう":"N5","あした":"N5","きのう":"N5","いま":"N5",
  "とき":"N5","なつ":"N5","ふゆ":"N5","はる":"N5","あき":"N5",
  "ほん":"N5","えんぴつ":"N5","かみ":"N5","かばん":"N5","くつ":"N5",
  "みず":"N5","おちゃ":"N5","たべもの":"N5","のみもの":"N5","くるま":"N5",
  "でんしゃ":"N5","はな":"N5","き":"N5","やま":"N5","うみ":"N5",
  "いく":"N5","くる":"N5","かえる":"N5","のむ":"N5","たべる":"N5",
  "みる":"N5","きく":"N5","よむ":"N5","かく":"N5","はなす":"N5",
  "おしえる":"N5","ならう":"N5","はたらく":"N5","やすむ":"N5","おきる":"N5",
  "ねる":"N5","する":"N5","ある":"N5","いる":"N5","なる":"N5",
  "わかる":"N5","しる":"N5","おもう":"N5","いう":"N5","もつ":"N5",
  "つかう":"N5","あける":"N5","しめる":"N5","だす":"N5","いれる":"N5",
  "まつ":"N5","かう":"N5","うる":"N5","くれる":"N5","あげる":"N5",
  "みせる":"N5","きめる":"N5","はじまる":"N5","おわる":"N5","つく":"N5",
  // ── N4 ───────────────────────────────────────────────────────────────────
  "こころ":"N4","きもち":"N4","こえ":"N4","すがた":"N4","かぜ":"N4",
  "ゆめ":"N4","ひかり":"N4","そら":"N4","つき":"N4","ほし":"N4",
  "かわ":"N4","もり":"N4","みち":"N4","とり":"N4","さくら":"N4",
  "あい":"N4","こい":"N4","とも":"N4","なかま":"N4","かぞく":"N4",
  "せかい":"N4","みらい":"N4","かこ":"N4","むかし":"N4",
  "じかん":"N4","ばしょ":"N4","ところ":"N4","おもいで":"N4","きぼう":"N4",
  "ゆき":"N4","あめ":"N4","かみなり":"N4","たいよう":"N4","つち":"N4",
  "いのち":"N4","ちから":"N4","からだ":"N4","て":"N4","め":"N4",
  "みえる":"N4","きこえる":"N4","かんじる":"N4","おぼえる":"N4","わすれる":"N4",
  "かわる":"N4","つたえる":"N4","まもる":"N4","さがす":"N4","あつまる":"N4",
  "はなれる":"N4","むかう":"N4","すすむ":"N4","のこる":"N4",
  "あう":"N4","いきる":"N4","しぬ":"N4","たたかう":"N4","まける":"N4",
  "かつ":"N4","うごく":"N4","とまる":"N4","はしる":"N4","とぶ":"N4",
  "うたう":"N4","えがく":"N4","かなしい":"N4","うれしい":"N4","さびしい":"N4",
  "たのしい":"N4","こわい":"N4","くるしい":"N4","やさしい":"N4","つらい":"N4",
  "あたたかい":"N4","つめたい":"N4","うつくしい":"N4","すばらしい":"N4",
  "ちいさな":"N4","おなじ":"N4","ちがう":"N4","たとえば":"N4","もちろん":"N4",
  // ── N3 (common lyric vocabulary) ─────────────────────────────────────────
  "かがやく":"N3","ひびく":"N3","ながれる":"N3","ひろがる":"N3","しずむ":"N3",
  "もえる":"N3","こぼれる":"N3","ちる":"N3","さく":"N3","かれる":"N3",
  "ふるえる":"N3","かがやき":"N3","めざめる":"N3","ふみだす":"N3","たどりつく":"N3",
  "えいえん":"N3","むげん":"N3","かなた":"N3","このよ":"N3","うちゅう":"N3",
  "かなしみ":"N3","よろこび":"N3","いかり":"N3","おそれ":"N3","のぞみ":"N3",
  "ちかい":"N3","ちかう":"N3","うしなう":"N3","てにいれる":"N3",
  "ひとり":"N3","ふたり":"N3","いっしょ":"N3","ならんで":"N3","つながる":"N3",
};

// ── Jōyō kanji difficulty tiers (for heuristic) ────────────────────────────────
// Grade 1-2 kanji appear in N5/N4; grade 7+ (secondary school) tend toward N2/N1.
const EASY_KANJI = new Set([..."日一国人年大十二本中長出三時行見月後前合東高間明"]);
const HARD_KANJI = new Set([..."憧憬憂鬱欝儚謳醸誰懸劫乙旋律奏魂彷徨漂漣煌綺羅纏"]);

// ── Public API ─────────────────────────────────────────────────────────────────

/**
 * Look up the JLPT level for a word.
 *
 * @param surface  Word as it appears (kanji/kana)
 * @param reading  Hiragana reading (optional but improves accuracy)
 * @param baseForm Dictionary form (optional)
 */
export function lookupJlpt(
  surface:  string,
  reading?: string,
  baseForm?: string,
): JlptLevel | null {
  // Try hiragana forms of the word
  const hiraReading  = reading  ? kataToHira(reading.toLowerCase())  : null;
  const hiraBase     = baseForm ? kataToHira(baseForm.toLowerCase())  : null;
  const hiraSurface  = kataToHira(surface.toLowerCase());

  for (const key of [hiraReading, hiraBase, hiraSurface].filter(Boolean)) {
    const level = JLPT_TABLE[key!];
    if (level) return level;
  }

  // Heuristic based on character difficulty
  return jlptHeuristic(surface);
}

function jlptHeuristic(surface: string): JlptLevel | null {
  if (!surface) return null;
  if (/^[ぁ-ん]+$/.test(surface)) return null;      // pure hiragana — likely particle/aux
  if (/^[ァ-ヶー]+$/.test(surface)) return null;     // pure katakana (loanword) — skip
  if (/^[a-zA-Z]+$/.test(surface)) return null;       // Latin

  let hardCount = 0;
  let easyCount = 0;
  for (const c of surface) {
    if (HARD_KANJI.has(c)) hardCount++;
    else if (EASY_KANJI.has(c)) easyCount++;
  }

  if (hardCount > 0) return "N1";
  if (easyCount === surface.replace(/[^一-鿿]/g, "").length && easyCount > 0) return "N4";
  return null;
}

/**
 * Given a list of JLPT levels, return the most advanced one.
 * (N1 > N2 > N3 > N4 > N5)
 */
export function mostAdvancedJlpt(levels: (JlptLevel | null)[]): JlptLevel | null {
  const order: JlptLevel[] = ["N1", "N2", "N3", "N4", "N5"];
  for (const lvl of order) {
    if (levels.some((l) => l === lvl)) return lvl;
  }
  return null;
}
