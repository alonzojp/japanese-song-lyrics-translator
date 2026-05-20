import type { Token } from "@japanese-lyrics/shared";

const JISHO_BASE = "https://jisho.org/api/v1/search/words?keyword=";
const CORS_PROXIES = [
  (url: string) => `https://corsproxy.io/?${url}`,
  (url: string) => `https://api.allorigins.win/raw?url=${encodeURIComponent(url)}`,
];

export async function lookupJisho(word: string): Promise<Token | null> {
  const targetUrl = JISHO_BASE + encodeURIComponent(word);

  for (const proxy of CORS_PROXIES) {
    try {
      const res = await fetch(proxy(targetUrl), {
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) continue;

      const data = (await res.json()) as {
        data?: Array<{
          japanese?: Array<{ reading?: string }>;
          senses?: Array<{ parts_of_speech?: string[]; english_definitions?: string[] }>;
          jlpt?: string[];
        }>;
      };

      if (!data.data?.length) continue;
      const e = data.data[0]!;

      return {
        surface: word,
        reading: e.japanese?.[0]?.reading ?? null,
        furigana: null,
        romaji: null,
        part_of_speech: e.senses?.[0]?.parts_of_speech?.[0] ?? null,
        dictionary_form: null,
        english: e.senses?.[0]?.english_definitions?.slice(0, 3).join("; ") ?? null,
        jlpt: (e.jlpt?.[0]?.replace("jlpt-n", "N") as Token["jlpt"]) ?? null,
        notes: null,
      };
    } catch {
      continue;
    }
  }

  return null;
}
