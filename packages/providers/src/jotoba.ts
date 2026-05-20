import type { Token } from "@japanese-lyrics/shared";

const JOTOBA_URL = "https://jotoba.de/api/search/words";

export async function lookupJotoba(word: string): Promise<Token | null> {
  const res = await fetch(JOTOBA_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: word, language: "English", no_english: false }),
  });

  if (!res.ok) throw new Error("jotoba fetch failed");

  const data = (await res.json()) as {
    words?: Array<{
      reading?: { kana?: string };
      senses?: Array<{ pos?: string[]; glosses?: string[] }>;
      jlpt_lvl?: number | null;
    }>;
  };

  if (!data.words?.length) return null;
  const w = data.words[0]!;

  return {
    surface: word,
    reading: w.reading?.kana ?? null,
    furigana: null,
    romaji: null,
    part_of_speech: w.senses?.[0]?.pos?.[0] ?? null,
    dictionary_form: null,
    english: w.senses?.[0]?.glosses?.slice(0, 3).join("; ") ?? null,
    jlpt: w.jlpt_lvl != null ? (`N${w.jlpt_lvl}` as Token["jlpt"]) : null,
    notes: null,
  };
}
