const COMBO: Record<string, string> = {
  きゃ: "kya", きゅ: "kyu", きょ: "kyo",
  しゃ: "sha", しゅ: "shu", しょ: "sho",
  ちゃ: "cha", ちゅ: "chu", ちょ: "cho",
  にゃ: "nya", にゅ: "nyu", にょ: "nyo",
  ひゃ: "hya", ひゅ: "hyu", ひょ: "hyo",
  みゃ: "mya", みゅ: "myu", みょ: "myo",
  りゃ: "rya", りゅ: "ryu", りょ: "ryo",
  ぎゃ: "gya", ぎゅ: "gyu", ぎょ: "gyo",
  じゃ: "ja",  じゅ: "ju",  じょ: "jo",
  びゃ: "bya", びゅ: "byu", びょ: "byo",
  ぴゃ: "pya", ぴゅ: "pyu", ぴょ: "pyo",
  ふぁ: "fa",  ふぃ: "fi",  ふぇ: "fe",  ふぉ: "fo",
};

const SINGLE: Record<string, string> = {
  あ: "a",  い: "i",  う: "u",  え: "e",  お: "o",
  か: "ka", き: "ki", く: "ku", け: "ke", こ: "ko",
  さ: "sa", し: "shi",す: "su", せ: "se", そ: "so",
  た: "ta", ち: "chi",つ: "tsu",て: "te", と: "to",
  な: "na", に: "ni", ぬ: "nu", ね: "ne", の: "no",
  は: "ha", ひ: "hi", ふ: "fu", へ: "he", ほ: "ho",
  ま: "ma", み: "mi", む: "mu", め: "me", も: "mo",
  や: "ya", ゆ: "yu", よ: "yo",
  ら: "ra", り: "ri", る: "ru", れ: "re", ろ: "ro",
  わ: "wa", を: "wo", ん: "n",
  が: "ga", ぎ: "gi", ぐ: "gu", げ: "ge", ご: "go",
  ざ: "za", じ: "ji", ず: "zu", ぜ: "ze", ぞ: "zo",
  だ: "da", ぢ: "ji", づ: "zu", で: "de", ど: "do",
  ば: "ba", び: "bi", ぶ: "bu", べ: "be", ぼ: "bo",
  ぱ: "pa", ぴ: "pi", ぷ: "pu", ぺ: "pe", ぽ: "po",
  ゃ: "ya", ゅ: "yu", ょ: "yo",
  ー: "-",
};

export function toRomaji(s: string): string {
  if (!s) return "";
  // Katakana → hiragana
  s = s.replace(/[ァ-ヶ]/g, (c) => String.fromCharCode(c.charCodeAt(0) - 0x60));

  let out = "";
  let i = 0;
  while (i < s.length) {
    if (s[i] === "っ") {
      const next = COMBO[s.slice(i + 1, i + 3)] ?? SINGLE[s[i + 1] ?? ""] ?? "";
      out += next[0] ?? "";
      i++;
      continue;
    }
    const two = s.slice(i, i + 2);
    if (COMBO[two]) {
      out += COMBO[two];
      i += 2;
    } else if (SINGLE[s[i] ?? ""]) {
      out += SINGLE[s[i]!];
      i++;
    } else {
      out += s[i];
      i++;
    }
  }
  return out;
}
