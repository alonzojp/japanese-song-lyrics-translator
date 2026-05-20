import type { Token } from "@japanese-lyrics/shared";
import { toRomaji, buildRubyHTML, looksLikeReading, KANJI_RE } from "@japanese-lyrics/japanese-processing";

const JLPT_COLORS: Record<string, string> = {
  N5: "#5cb85c",
  N4: "#5bc0de",
  N3: "#7986cb",
  N2: "#ba68c8",
  N1: "#e05252",
};

const BADGE_BASE =
  "border-radius:4px;padding:2px 10px;font-size:0.72em;font-weight:700;letter-spacing:0.07em;margin:0 3px;";

const CARD_CSS = `<style>
.af,.ab{font-family:-apple-system,'Hiragino Sans','Noto Sans JP',sans-serif;}
.af{text-align:center;padding:44px 20px 32px;}
.af-w{font-size:3.6em;font-weight:700;line-height:1.1;letter-spacing:0.02em;}
.ab{max-width:500px;margin:0 auto;padding:20px 16px;}
.ab-word{font-size:2em;font-weight:700;text-align:center;margin-bottom:6px;line-height:2.4;}
.ab-word rt{font-size:0.4em;text-align:center;}
.ab-bdg{text-align:center;margin-bottom:14px;}
.ab-def{font-size:1.55em;font-weight:600;text-align:center;margin-bottom:8px;line-height:1.3;}
.ab-note{font-size:0.88em;color:#888;font-style:italic;text-align:center;margin-bottom:14px;}
.ab-hr{border:none;border-top:1px solid currentColor;opacity:0.15;margin:0 0 16px;}
.ab-ex{border-left:3px solid #7986cb;padding:10px 14px;margin-top:8px;}
.ab-exl{font-size:0.65em;color:#888;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px;font-weight:600;}
.ab-ext{font-size:1.05em;line-height:2.4;}
.ab-ext rt{font-size:0.5em;text-align:center;}
.ab-en{font-size:0.88em;color:#888;font-style:italic;margin-top:8px;line-height:1.5;}
@media(prefers-color-scheme:dark){.af-w,.ab-word,.ab-def,.ab-ext{color:#eeeeee;}}
.night_mode .af-w,.night_mode .ab-word,.night_mode .ab-def,.night_mode .ab-ext{color:#eeeeee;}
</style>`;

export interface AnkiCardContext {
  sentence?: string;
  sentenceRuby?: string;
  naturalTranslation?: string;
  profile?: string;
  deck?: string;
}

export function buildAnkiFront(token: Token): string {
  const dictWord = token.dictionary_form ?? token.surface;
  return CARD_CSS + `<div class="af"><div class="af-w">${dictWord}</div></div>`;
}

export function buildAnkiBack(token: Token, ctx: AnkiCardContext = {}): string {
  const dictWord = token.dictionary_form ?? token.surface;
  const reading = token.reading && token.reading !== token.surface ? token.reading : "";
  const romaji = token.romaji ?? toRomaji(token.reading ?? "");
  const def = token.english && !looksLikeReading(token.english) ? token.english : "";

  const jlptColor = token.jlpt ? JLPT_COLORS[token.jlpt] : null;
  const jlptBadge = jlptColor
    ? `<span style="${BADGE_BASE}border:1.5px solid ${jlptColor};color:${jlptColor};">${token.jlpt}</span>`
    : "";
  const posBadge = token.part_of_speech
    ? `<span style="${BADGE_BASE}border:1.5px solid #7986cb;color:#7986cb;text-transform:uppercase;letter-spacing:0.1em;">${token.part_of_speech}</span>`
    : "";

  // Apply ruby only when the surface form is shown (not dictionary form)
  const wordRuby =
    reading && KANJI_RE.test(dictWord) && dictWord === token.surface
      ? buildRubyHTML(dictWord, reading, token.furigana) ??
        `<ruby>${dictWord}<rt>${reading}</rt></ruby>`
      : dictWord;

  let back = CARD_CSS + `<div class="ab">`;
  back += `<div class="ab-word">${wordRuby}</div>`;

  if (posBadge || jlptBadge) back += `<div class="ab-bdg">${posBadge}${jlptBadge}</div>`;
  if (def) back += `<div class="ab-def">${def}</div>`;
  if (token.notes) back += `<div class="ab-note">${token.notes}</div>`;
  if (romaji && !def) back += `<div class="ab-note">${romaji}</div>`;

  if (ctx.sentence) {
    const sentRuby = ctx.sentenceRuby ?? ctx.sentence;
    back +=
      `<hr class="ab-hr">` +
      `<div class="ab-ex">` +
      `<div class="ab-exl">Example</div>` +
      `<div class="ab-ext">${sentRuby}</div>` +
      (ctx.naturalTranslation ? `<div class="ab-en">${ctx.naturalTranslation}</div>` : "") +
      `</div>`;
  }

  back += `</div>`;
  return back;
}
