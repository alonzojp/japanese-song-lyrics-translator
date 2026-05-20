import type { Token } from "@japanese-lyrics/shared";
import { lookupJotoba } from "./jotoba.js";
import { lookupJisho } from "./jisho.js";

export async function basicLookup(word: string): Promise<Token | null> {
  try {
    const result = await lookupJotoba(word);
    if (result) return result;
  } catch {}

  try {
    const result = await lookupJisho(word);
    if (result) return result;
  } catch {}

  return null;
}
