/**
 * Dictionary service for English definitions + JLPT data.
 *
 * Interface-first design — callers depend on DictionaryService,
 * not on any particular backend.
 *
 * Default implementation: JotobaDictionaryService
 *   - Primary: Jotoba API (https://jotoba.de)
 *   - Aggressive LRU caching (3000 entries, process-lifetime)
 *   - Batch support: fetches up to BATCH_SIZE words in parallel
 *   - Graceful degradation: returns null on network failure
 */
import type { JlptLevel } from "./models.js";
import { LRUCache } from "./cache.js";

// ── Interface ──────────────────────────────────────────────────────────────────

export interface DictionaryEntry {
  surface:  string;
  reading:  string | null;
  meanings: string[];            // all English definitions
  jlpt:     JlptLevel | null;
  pos:      string[];            // parts of speech
}

export interface DictionaryService {
  lookup(word: string, reading?: string): Promise<DictionaryEntry | null>;
  lookupBatch(words: string[]): Promise<Map<string, DictionaryEntry | null>>;
}

// ── Jotoba implementation ──────────────────────────────────────────────────────

const JOTOBA_URL    = "https://jotoba.de/api/search/words";
const BATCH_SIZE    = 8;     // parallel requests per batch
const CACHE_MAX     = 3000;

interface JotobaWord {
  reading?: { kana?: string };
  senses?:  Array<{ pos?: string[]; glosses?: string[] }>;
  jlpt_lvl?: number | null;
}

interface JotobaResponse {
  words?: JotobaWord[];
}

export class JotobaDictionaryService implements DictionaryService {
  private readonly cache = new LRUCache<string, DictionaryEntry | null>(CACHE_MAX);

  async lookup(word: string, _reading?: string): Promise<DictionaryEntry | null> {
    if (!word.trim()) return null;

    const cached = this.cache.get(word);
    if (cached !== undefined) return cached;

    const result = await this._fetch(word);
    this.cache.set(word, result);
    return result;
  }

  async lookupBatch(words: string[]): Promise<Map<string, DictionaryEntry | null>> {
    const result = new Map<string, DictionaryEntry | null>();
    const missing = words.filter((w) => !this.cache.has(w));

    // Return cached entries first
    for (const w of words) {
      const cached = this.cache.get(w);
      if (cached !== undefined) result.set(w, cached);
    }

    // Fetch missing words in parallel batches
    for (let i = 0; i < missing.length; i += BATCH_SIZE) {
      const batch   = missing.slice(i, i + BATCH_SIZE);
      const entries = await Promise.all(batch.map((w) => this.lookup(w)));
      for (let j = 0; j < batch.length; j++) {
        result.set(batch[j]!, entries[j]!);
      }
    }

    return result;
  }

  private async _fetch(word: string): Promise<DictionaryEntry | null> {
    try {
      const res = await fetch(JOTOBA_URL, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ query: word, language: "English", no_english: false }),
      });
      if (!res.ok) return null;

      const data = (await res.json()) as JotobaResponse;
      if (!data.words?.length) return null;

      const w = data.words[0]!;
      return {
        surface:  word,
        reading:  w.reading?.kana ?? null,
        meanings: w.senses?.flatMap((s) => s.glosses ?? []).slice(0, 5) ?? [],
        jlpt:     w.jlpt_lvl != null ? (`N${w.jlpt_lvl}` as JlptLevel) : null,
        pos:      w.senses?.[0]?.pos ?? [],
      };
    } catch {
      return null;
    }
  }
}

// ── Singleton default service ──────────────────────────────────────────────────

let _defaultService: DictionaryService | null = null;

export function getDefaultDictionaryService(): DictionaryService {
  if (!_defaultService) _defaultService = new JotobaDictionaryService();
  return _defaultService;
}

export function setDictionaryService(service: DictionaryService): void {
  _defaultService = service;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

/** Extract the primary English meaning from an entry. */
export function primaryMeaning(entry: DictionaryEntry | null): string | null {
  return entry?.meanings[0] ?? null;
}
