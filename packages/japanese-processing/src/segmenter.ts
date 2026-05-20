const JAPANESE_RE = /[぀-ヿ㐀-䶿一-鿿ｦ-ﾟ]+/g;

export function segmentText(text: string): string[] {
  if (typeof Intl !== "undefined" && "Segmenter" in Intl) {
    const seg = new Intl.Segmenter("ja", { granularity: "word" });
    return [...seg.segment(text)]
      .filter((s) => s.isWordLike)
      .map((s) => s.segment);
  }
  return text.match(JAPANESE_RE) ?? [];
}
