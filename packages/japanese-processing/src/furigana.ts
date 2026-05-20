import type { Token } from "@japanese-lyrics/shared";

export const KANJI_RE = new RegExp(
  "[" +
    String.fromCodePoint(0x4e00) +
    "-" +
    String.fromCodePoint(0x9fff) +
    String.fromCodePoint(0x3400) +
    "-" +
    String.fromCodePoint(0x4dbf) +
    "]"
);

const CJK_CLASS =
  String.fromCodePoint(0x4e00) +
  "-" +
  String.fromCodePoint(0x9fff) +
  String.fromCodePoint(0x3400) +
  "-" +
  String.fromCodePoint(0x4dbf) +
  String.fromCodePoint(0x3005) +
  String.fromCodePoint(0x3006);

const OPEN = "[(\\uff08]";
const CLOSE = "[)\\uff09]";
const RUBY_RE = new RegExp("([" + CJK_CLASS + "]+)" + OPEN + "([^)\\uff09]+)" + CLOSE, "g");
const STRIP_RE = new RegExp(OPEN + "[^)\\uff09]+" + CLOSE, "g");

export function looksLikeReading(s: string): boolean {
  return /^[぀-ゟ゠-ヿ\s・ー]+$/.test((s ?? "").trim());
}

/** Converts "食(た)べる" API format → `<ruby>食<rt>た</rt></ruby>べる`. */
export function furiganaToRuby(text: string): string {
  if (!text) return "";
  let out = text.replace(RUBY_RE, "<ruby>$1<rt>$2</rt></ruby>");
  out = out.replace(STRIP_RE, "");
  return out;
}

/**
 * Strips matching kana suffix so ruby only wraps the kanji part.
 * e.g. surface=始めて reading=はじめて → "始(はじ)めて"
 */
export function alignFurigana(surface: string, reading: string): string | null {
  if (!reading || reading === surface || !KANJI_RE.test(surface)) return null;

  let si = surface.length - 1;
  let ri = reading.length - 1;

  while (
    si > 0 &&
    ri > 0 &&
    surface[si] === reading[ri] &&
    /[ぁ-んァ-ン]/.test(surface[si]!)
  ) {
    si--;
    ri--;
  }

  const kanjiPart = surface.slice(0, si + 1);
  const kanaSuffix = surface.slice(si + 1);
  const kanjiRead = reading.slice(0, ri + 1);

  if (!kanjiPart || !kanjiRead || kanjiRead === kanjiPart) return null;
  return kanjiPart + "(" + kanjiRead + ")" + kanaSuffix;
}

/**
 * Returns `<ruby>` HTML for a surface/reading pair.
 * Prefers per-kanji furigana annotation, falls back to kana-suffix stripping,
 * then whole-word ruby as last resort.
 */
export function buildRubyHTML(
  surface: string,
  reading: string,
  furigana?: string | null
): string | null {
  if (!KANJI_RE.test(surface)) return null;
  const furi = furigana ?? alignFurigana(surface, reading);
  if (furi) return furiganaToRuby(furi);
  if (reading && reading !== surface)
    return `<ruby>${surface}<rt>${reading}</rt></ruby>`;
  return null;
}

/** Builds complete sentence ruby HTML from a token array. */
export function buildRubyFromTokens(tokens: Token[]): string {
  return tokens
    .map(
      (t) =>
        buildRubyHTML(t.surface ?? "", t.reading ?? "", t.furigana) ??
        (t.surface ?? "")
    )
    .join("");
}
